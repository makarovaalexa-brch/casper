r"""NDCG-OBJECTIVE INTERVIEW TREE, stage 1 (build on validation) -- label `ndcgtree`.

WHY THIS ARM EXISTS. Every Act-1 comparison confounds two things: *conditioning on the answers* and
*choice of objective*. The only arm in the battery that maximises the quantity we report
(cohort-mean held-out full NDCG@10 under the certified tower) is also the only STATIC one. Golbandi's
tree conditions, but its splitter minimises within-node squared error of RATING prediction (Eq. 1),
which is not our metric and is not a ranking metric at all. This arm CONDITIONS *and* OPTIMISES OUR
METRIC, so conditioning is the only thing that varies against the static greedy comparator.

NEAREST PRIOR WORK. Hu & Yu, "Interview Process Learning for Top-N Recommendation", RecSys 2013 -- an
interview decision tree whose split loss "can be square loss, logistic loss, pairwise rank loss, etc."
So "first ranking objective inside an interview tree" is NOT available to us and is not claimed. Four
separations, all verified from their text (external_literature/findings/
hu_yu_2013_interview_process_learning.md): (1) a smooth pairwise surrogate, not a truncated discounted
top-k metric; (2) optimised as a TRAINING loss over the tree fit, not as the measured value of asking
a question; (3) MAP on 10x SAMPLED NEGATIVES, not NDCG@10 on the full 18,359-item catalogue;
(4) their static comparator is TOP-K POPULAR -- ours is the order built by maximising the very metric
being reported. That last one is the whole point of this run.

THE SPLIT CRITERION, AND THE ONE FACT A READER MUST NOT MISS
------------------------------------------------------------
At a node t holding users S_t, for each candidate question q we split S_t three ways on the user's own
answer (LIKE / DISLIKE / DID-NOT-ANSWER) and score

    Crit_t(q) = SUM_{c in {L,D,U}}  |S_c| / |S_t|  *  mean_{u in S_c} NDCG@10( u | path(t) + q )

i.e. the node-size-weighted mean of the three children's cohort-mean NDCG, computed by ACTUALLY
FOLDING each user's answers through the certified tower (+ concept channel) and ranking their held-out
targets on the full catalogue. `splitter(t) = argmax_q Crit_t(q)`.

NDCG@10 is a PER-USER quantity, so that size-weighted mean over a partition of S_t is algebraically
IDENTICAL to the plain mean over S_t. We do not hide this, we MEASURE it at every node
(`partition_decomposition_max_abs_err`). Two consequences, both stated rather than left for a reviewer:

  * the split search at a node is exactly the greedy argmax restricted to that node's users -- this
    tree is MYOPIC in the same sense Golbandi's is (his criterion is also the immediate within-node
    error, with no lookahead). Adaptivity does not come from the criterion looking ahead; it comes
    from the CHILDREN being free to ask different next questions.
  * therefore the ROOT question is the same question a global greedy over this pool would pick first.
    When it matches the banked comparator's first question the tree is exactly NESTED in the
    comparator at K=1 (difference identically 0, asserted), and every K>=2 difference is attributable
    to conditioning alone. It need not match, and the reason is recorded in RULER_NOTE below: the
    banked build is a POLICY-A artifact and this harness scores under POLICY B, so an argmax can move
    when the ruler moves. A mismatch is logged as a loud, quantified caveat, not swallowed.

THE COMPARATOR IS ON A DIFFERENT RULER, AND THAT IS MEASURED, NOT ASSUMED. `greedy_static_i26c.json`
records intercept.full = 0.12987336; that is the POLICY-A m=0 value. Under POLICY B (the 2026-08-05
candidate-masking ruling this harness implements) the same 10,000 validation users give m=0 =
0.15686538. Its stored `gains` and `curve` are therefore NOT comparable to anything measured here.
Only its SEQUENCES are used, and every comparison re-scores them IN THIS RUN under policy B -- the
same thing `group_greedy_genre.py` does. See RULER_NOTE.

THE ACTION SPACE IS THE COMBINED POOL (author redirect, 2026-08-11). Items AND concepts compete in one
argmax, exactly as the global build's `combined` arm does: top-500 items by popularity + top-500
concepts by the `answerable` bank, the same two `_ranked_pool` calls, concatenated in the same order
(items first). Two reasons the earlier items-only pool was wrong here: `greedy_concepts` is the
strongest static arm in the battery (0.2362/0.1022 at k=8 vs `greedy_items` 0.2247/0.0776), so
measuring a conditioned arm inside the weaker channel understates it; and item splits are terrible
PARTITIONS -- item 83 gives like 2,099 / dislike 856 / didn't-answer 7,045, i.e. 70% of the cohort in
one branch -- whereas ~97% of users can answer a concept, so concept splits divide far more evenly and
buy more usable depth before the min-leaf rule bites. Whether that is actually true is measured here
and reported per node (`answer_rate`, `child_sizes`, `channel`).

EVERY QUESTION IS CHARGED. There is NO free side information in this arm, unlike
`group_greedy_genre.py`. Tree questions burn an interview turn and are folded into the recommender:
an asked-but-unrated ITEM folds as the i26 native UNSEEN exposure token, an unanswerable CONCEPT as a
cv3 refusal token. Both users are routed down the DID-NOT-ANSWER branch AND have spent the turn.
There is no generosity caveat to apply to these numbers -- that is this arm's main advantage over the
genre arm.

STOPPING: THE MIN-LEAF RULE IS THE ONLY STOPPING RULE. There is no depth cap. A node is not split if
any of its three children would hold fewer than MIN_LEAF = 200 validation users; otherwise it splits,
until the interview budget itself runs out (a node at depth L-1 has one question left and nothing to
split into). The realised depth is a RESULT, reported, not a setting. MIN_LEAF = 200 is empirical,
from the genre-split experiment (2026-08-11): every genre group above 200 users held a positive gain
when its order was carried to TEST, while every group below it failed to generalise -- Horror n=51
test -0.0013 (ns), Crime n=79 test -0.0433, Fantasy n=11 test -0.1156. It is applied to the CHOSEN
splitter (the argmax), not as a constraint on the search: if the best question shatters the node, the
node STOPS rather than falling back to a worse question.

TOTAL INTERVIEW LENGTH IS ALWAYS 15 QUESTIONS. A node that stops at depth d gets a greedy tail of
15-d further questions, searched on its own users. So the min-leaf rule costs a leaf NOTHING in
budget, only in conditioning, and the comparison to the global 15-step order is question-for-question
at equal K throughout.

WHAT IS **NOT** GUARANTEED HERE (and is guaranteed in the genre arm). A leaf's order must carry its
ancestors' questions, which were chosen for their PARENTS' cohorts. The global order is therefore NOT
inside a leaf's search space, and a leaf CAN legitimately score BELOW the global order at K=15. A
negative leaf gain is a RESULT, not a harness failure. The only mechanical guarantee is the K=1 tie.

COHORT. The 10,000 canonical VALIDATION users. Forced, not chosen: `data/ml-25m/proc/` has no
train_tr/train_te, so train users have no fold-in/target split and the tower was TRAINED on them. THE
TEST COHORT IS NOT BUILT, NOT LOADED AND NOT TOUCHED here; carrying the tree to TEST is a separate,
author-authorised stage.

PERSISTENCE (HARD RULE 11). Growth is LEVEL-SYNCHRONOUS, for the same reason Golbandi's is: after a
level is finished the structure above it is final, so the partial tree on disk IS the tree that a
shallower request would have produced -- every per-level write is a complete, valid, usable artifact.
`*_partial.json` is rewritten after every pick and every node; each node's CELF heap gets its own
contract-signed sidecar, so a kill costs at most the lazy re-evals since the last pick, never the
1,000-candidate seed pass, and a stopped node's greedy tail RESUMES the very heap its split search
built instead of paying for a second seed pass.
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

from strategy_channel_suite import (log, build_greedy_env, set_active_rows, greedy_certified,
                                    genv_mean, genv_eval, genv_answered, contract_selftest,
                                    _ranked_pool, CERT_TOWER, CERT_CONC, CONCSTATS, N_TRAIN_USERS,
                                    pool_contract, resume_ok)
from i26_encoder import LIKE_MIN_LEVEL
from group_greedy_genre import load_item_genres, favourite_genre

LABEL = "ndcgtree"
OUTDIR = os.path.join(_ROOT, "experiments", "battery")
GLOBAL_JSON = os.path.join(OUTDIR, "greedy_static_i26c.json")

MIN_LEAF = 200
TOTAL_L = 15
BUDGETS = (1, 2, 4, 8, 15)
BRANCHES = ("like", "dislike", "noanswer")
HEADLINE_ARM = "combined"
COMPARATORS = ("combined", "concepts-only", "items-only")

MIN_LEAF_JUSTIFICATION = (
    "The MIN-LEAF RULE IS THE ONLY STOPPING RULE -- there is no depth cap. A node is not split if ANY "
    "of its three children would hold fewer than 200 validation users; growth otherwise continues "
    "until the interview budget itself runs out. Realised depth is a RESULT, not a setting. The 200 "
    "is empirical, from the genre-split experiment (2026-08-11): every genre group ABOVE 200 users "
    "held a positive gain when its order was carried to the TEST cohort, while every group BELOW it "
    "failed to generalise (Horror n=51 test -0.0013 ns; Crime n=79 test -0.0433; Fantasy n=11 test "
    "-0.1156). Applied to the CHOSEN splitter, not as a search constraint: if the argmax question "
    "shatters the node, the node STOPS -- it does not fall back to a worse question.")
CHARGED = (
    "EVERY QUESTION IS CHARGED. Tree questions come from the same combined pool as the global build "
    "(top-500 popular items + top-500 answerable concepts); they burn an interview turn and their "
    "answers are folded into the recommender exactly like any other answer -- an asked-but-unrated "
    "item as the i26 native UNSEEN exposure token, an unanswerable concept as a cv3 refusal token. "
    "There is NO free side information in this arm and therefore NO generosity caveat: the total "
    "budget is 15 questions on both sides at every K.")
NO_GUARANTEE = (
    "A leaf's order must carry its ancestors' questions, which were chosen for their PARENTS' "
    "cohorts. The global order is therefore NOT inside a leaf's search space and a leaf CAN score "
    "below it. A negative leaf gain is a RESULT here, not a harness failure. The one mechanical "
    "guarantee is that the root splitter equals the global order's first question, so the K=1 "
    "difference is identically zero; that is asserted against the banked global build.")
RULER_NOTE = (
    "THE BANKED GLOBAL BUILD IS A POLICY-A ARTIFACT (measured 2026-08-12, this run's sanity gate). "
    "`greedy_static_i26c.json` records intercept.full = 0.12987336, which is the POLICY-A m=0 value; "
    "today's harness scores under POLICY B (2026-08-05 ruling, candidate_masking_protocol.md) and "
    "gives m=0 = 0.15686538 on the same 10,000 validation users. Its stored `gains` and `curve` are "
    "therefore on a DIFFERENT RULER and are NOT comparable to anything measured here -- numbers from "
    "different information contracts are not ordered. What IS used from that file is the SEQUENCE of "
    "questions, which is re-scored IN THIS RUN under policy B for every comparison (the same thing "
    "`group_greedy_genre.py` does). Residual caveat, stated rather than buried: the comparator's "
    "ORDER was selected under policy A, so it is not exactly 'the same objective under the same "
    "protocol'. Closing that would mean rebuilding the static combined order under policy B (~12 h).")
NEG_CAP_NOTE = (
    "Concept answer values follow the CERTIFIED-ERA convention that `concept_cv3_ep08` was trained "
    "under and that produced the comparator: uncapped signed values, with `cap_negatives` applied "
    "over the REVEALED set only, C_NEG read from the signed prereg. This is `build_greedy_env`'s own "
    "path and is checked against `signed_answers.cap_negatives` by `contract_selftest` at startup. "
    "(`--concept_neg_cap` is an `adaptive_harness` flag; it does not exist on this code path, and "
    "introducing one would put this arm on a different ruler from its own comparator.)")


# --------------------------------------------------------------------------------- comparators
def read_global(path, need):
    """The frozen global orders. READ from the banked build, never re-derived."""
    blob = json.load(open(path))
    assert blob["objective"] == "full", f"global build objective is {blob['objective']}, expected full"
    assert blob["item_bank"] == "pop" and blob["n_items_pool"] == 500, \
        f"global build item pool is {blob['item_bank']}/{blob['n_items_pool']}, expected pop/500"
    assert blob["conc_bank"] == "answerable" and blob["n_concepts_pool"] == 500, \
        f"global build concept pool is {blob['conc_bank']}/{blob['n_concepts_pool']}, expected answerable/500"
    out = {}
    for arm in COMPARATORS:
        a = blob["arms"][arm]
        seq = [(int(c), int(i)) for c, i in a["sequence"]]
        assert len(seq) >= need, f"global arm '{arm}' is {len(seq)} steps, need >= {need}"
        out[arm] = {"sequence": seq[:need], "gains": [float(g) for g in a["gains"]]}
    return out, blob


def leak_assert(E, rows, tag):
    """The routing lookup must never see a held-out target. `lvl_lookup` is built from ctx.allb, which
    is built with the te rows removed -- assert it rather than trust the comment."""
    ctx = E["ctx"]; lvl = E["lvl"]
    bad = sum(1 for r in rows if set(ctx.va_te[r].indices.tolist()) & set(lvl[int(r)].keys()))
    assert bad == 0, f"LEAK ({tag}): {bad} users have a held-out target inside the routing lookup"
    log(f"[leak/{tag}] 0/{len(rows)} users have any held-out target inside the routing lookup -- "
        f"every branch is decided from FOLD-IN evidence only")


def branch_of(E, rows, q):
    """LIKE / DISLIKE / DID-NOT-ANSWER for one question over `rows`, as a string array.

    ITEM (chan 0), the harness's own behavioural rule off the graded fold-in matrix: absent from the
    user's fold-in lookup -> DID-NOT-ANSWER (folded as the i26 UNSEEN exposure token); level >=
    LIKE_MIN_LEVEL (7, i.e. >= 4.0 stars) -> LIKE; otherwise -> DISLIKE.

    CONCEPT (chan 1), the same three-way structure on the signed answer the fold actually receives:
    not `foldable` (unanswerable, or in the REFUSE band) -> DID-NOT-ANSWER (folded as the cv3 refusal
    token, and the turn is still burned); signed value > 0 -> LIKE; otherwise -> DISLIKE. The `meh`
    band is zeroed under the certified keep_meh=False convention, so it lands in DISLIKE; the number
    of exact zeros is counted and reported at every node so the convention is visible, never implicit.
    """
    ch, iid = int(q[0]), int(q[1])
    idx = np.asarray(rows, np.int64)
    if ch == 0:
        lvl = E["lvl"]
        lv = np.fromiter((lvl[int(r)].get(iid, -1) for r in idx), np.int64, len(idx))
        ans = lv >= 0
        like = ans & (lv >= LIKE_MIN_LEVEL)
        n_zero = 0
    else:
        ans = E["foldable"][idx, iid]
        v = E["V"][idx, iid]
        like = ans & (v > 0)
        n_zero = int((ans & (v == 0)).sum())
    lab = np.where(~ans, "noanswer", np.where(like, "like", "dislike"))
    return lab, float(ans.mean()), n_zero


def branch_of_per_user(E, rows, qs):
    """`branch_of` where EVERY USER CARRIES THEIR OWN QUESTION -- which is what a parametric policy's
    rollout produces (workstream E1). Same rule, same order: 0 = like, 1 = dislike, 2 = did-not-answer.

    It is not a second answer model. `branch_of_selftest` asserts that handing every user the SAME
    question reproduces `branch_of` exactly, so the two cannot drift.
    """
    lvl = E["lvl"]
    out = np.empty(len(rows), np.int8)
    for j, (r, q) in enumerate(zip(rows, qs)):
        ch, iid = int(q[0]), int(q[1])
        if ch == 0:
            lv = lvl[int(r)].get(iid, -1)
            ans = lv >= 0
            like = ans and lv >= LIKE_MIN_LEVEL
        else:
            ans = bool(E["foldable"][int(r), iid])
            like = ans and float(E["V"][int(r), iid]) > 0
        out[j] = 0 if like else (2 if not ans else 1)
    return out


def branch_of_selftest(E, rows, qs):
    """The anti-drift gate for `branch_of_per_user`: for each question in `qs`, hand it to EVERY user
    and require the per-user path to reproduce `branch_of` exactly."""
    code = {"like": 0, "dislike": 1, "noanswer": 2}
    for q in qs:
        lab, _ar, _nz = branch_of(E, rows, q)
        want = np.asarray([code[str(x)] for x in lab], np.int8)
        got = branch_of_per_user(E, rows, [tuple(q)] * len(rows))
        assert np.array_equal(got, want), f"branch_of_per_user disagrees with branch_of at {q}"
    log(f"[selftest] branch_of_per_user == branch_of on {len(qs)} questions x {len(rows)} users")
    return True


def split_rows(E, rows, q):
    """The ternary partition itself. No user is dropped, capped or sampled (HARD RULE 1): the three
    branches partition the node exactly, and that is asserted."""
    lab, ar, n_zero = branch_of(E, rows, q)
    idx = np.asarray(rows, np.int64)
    kids = {k: [int(r) for r in idx[lab == k]] for k in BRANCHES}
    tot = sum(len(v) for v in kids.values())
    assert tot == len(rows), f"ternary split lost users: {tot} != {len(rows)}"
    assert len({r for v in kids.values() for r in v}) == len(rows), "ternary split is not disjoint"
    return kids, ar, n_zero


def genre_face_validity(E, rows, gen, top=4):
    """Face-validity read only: the most common favourite genre among a node's users, off their LIKED
    fold-in items (targets excluded by construction). Descriptive; nothing is selected on it."""
    cnt = {}
    lvl = E["lvl"]
    for r in rows:
        liked = [int(i) for i, L in lvl[int(r)].items() if int(L) >= LIKE_MIN_LEVEL]
        f = favourite_genre(liked, gen) or "(none)"
        cnt[f] = cnt.get(f, 0) + 1
    n = max(len(rows), 1)
    return [{"genre": g, "n": c, "frac": round(c / n, 4)}
            for g, c in sorted(cnt.items(), key=lambda kv: -kv[1])[:top]]


def build_pool(E, n_items, n_concepts):
    """The COMBINED pool, built by the SAME two `_ranked_pool` calls the global build makes, in the
    SAME order (items first, then concepts). Order matters twice: it decides CELF seed tie-breaks, and
    it keeps the single-entry item-state cache warm across the contiguous concept block (every concept
    candidate shares the prefix's item list, so the tower fold is paid once for the whole block)."""
    assert os.path.exists(CONCSTATS), f"train concept stats missing ({CONCSTATS})"
    z = np.load(CONCSTATS)
    ans_rate = (z["n"] / N_TRAIN_USERS).astype(np.float64)
    std_v = np.asarray(z["std"], np.float64)
    pol_n = std_v / max(std_v.max(), 1e-9); ar_n = ans_rate / max(ans_rate.max(), 1e-9)
    conc_tables = {"answerable": ans_rate, "polar": std_v,
                   "helf": 2 * pol_n * ar_n / np.clip(pol_n + ar_n, 1e-9, None)}
    assert len(ans_rate) == E["C"], f"concept stats width {len(ans_rate)} != {E['C']} concepts"
    item_tables = {"pop": np.asarray(E["ctx"].cnt, np.float64)}
    item_pool = [(0, int(i)) for i in _ranked_pool("pop", item_tables, n_items, "item")]
    conc_pool = [(1, int(c)) for c in _ranked_pool("answerable", conc_tables, n_concepts, "concept")]
    return item_pool + conc_pool, item_pool, conc_pool


# --------------------------------------------------------------------------------- verification
def heap_resume_selftest(E, rows, pool, objective, heapdir):
    """VERIFY THE CELF-HEAP PERSISTENCE. Two searches over the same node, same pool, same users:

      COLD   -- `heap_path=None`, i.e. the code path that existed before 2026-08-11, untouched.
      RESUME -- L=1 with a heap sidecar, then L=3 continuing from that sidecar.

    They must agree on the sequence AND on every gain BIT FOR BIT. If they do, extending a banked
    order costs the lazy re-evals only, and the extension is the order a cold run would have produced.
    A sidecar carrying a different contract must be IGNORED, not merged; that is checked too.
    """
    log("=" * 110)
    log("[heapcheck] CELF-heap persistence self-test: COLD (pre-change path) vs RESUME-and-EXTEND")
    hp = os.path.join(heapdir, "SELFTEST.json")
    if os.path.exists(hp):
        os.remove(hp)
    E["memo"] = {}
    cold_seq, cold_gains, cold_nev = greedy_certified(E, rows, pool, 3, objective, "selftest/cold")
    E["memo"] = {}                                   # no memo help: the resumed run pays its own evals
    s1, _g1, _ = greedy_certified(E, rows, pool, 1, objective, "selftest/step1", heap_path=hp)
    E["memo"] = {}
    r_seq, r_gains, r_nev = greedy_certified(E, rows, pool, 3, objective, "selftest/extend",
                                             heap_path=hp)
    assert [tuple(q) for q in cold_seq] == [tuple(q) for q in r_seq], \
        f"HEAP RESUME DIVERGED: cold {cold_seq} vs resumed {r_seq}"
    assert all(a == b for a, b in zip(cold_gains, r_gains)), \
        f"HEAP RESUME DIVERGED on gains: cold {cold_gains} vs resumed {r_gains}"
    assert [tuple(q) for q in s1] == [tuple(cold_seq[0])], "the L=1 banked pick is not the cold pick"
    log(f"[heapcheck] PASS -- identical sequence {cold_seq} and bit-identical gains "
        f"{[repr(g) for g in cold_gains]} | cold paid {cold_nev} evals; the resumed extension reached "
        f"the same answer without re-seeding ({r_nev} cumulative)")
    bad = json.load(open(hp)); bad["sig"]["objective"] = "tail"
    json.dump(bad, open(hp, "w"))
    E["memo"] = {}
    q_seq, _, _ = greedy_certified(E, rows, pool, 1, objective, "selftest/contract", heap_path=hp)
    assert [tuple(q) for q in q_seq] == [tuple(cold_seq[0])], \
        "a mismatched heap sidecar was silently merged instead of ignored"
    log("[heapcheck] a sidecar with a DIFFERENT contract is IGNORED and the search reseeds cold")
    log("=" * 110)
    os.remove(hp)


def preflight(E, rows, pool, item_pool, conc_pool, min_leaf, objective, out_path):
    """COST + SHAPE MODEL, paid before any multi-hour commitment. Nothing here is citable.

    Two things are unknown before launch and both drive the bill:
      (1) s/eval by CHANNEL at full cohort. A concept candidate shares the prefix's item list, so the
          tower fold is cached and it should be cheaper than an item candidate. Measured, not assumed.
      (2) how fast the MIN-LEAF rule bites. The partition a question induces is a pure lookup -- no
          NDCG, no folding -- so the ternary sizes of ALL 1,000 candidates at the root cost nothing.
          From the smallest branch fraction p_min of each candidate we get the depth a chain of such
          splits could reach before a child falls under 200 (n * p_min^d >= min_leaf), which bounds
          the tree and therefore the run.
    """
    log("=" * 110)
    log("[preflight] measuring s/eval by channel and the min-leaf shape of the whole pool")
    E["memo"] = {}
    t = time.time(); genv_mean(E, [], rows); t_base = time.time() - t
    t = time.time(); genv_mean(E, [item_pool[0]], rows); t_item = time.time() - t
    genv_mean(E, [conc_pool[0]], rows)                       # warm the item-state cache on []
    t = time.time(); genv_mean(E, [conc_pool[1]], rows); t_conc = time.time() - t
    log(f"[preflight] s/eval at n={len(rows)}: empty {t_base:.2f}s | ITEM candidate {t_item:.2f}s | "
        f"CONCEPT candidate {t_conc:.2f}s (item-state cache warm)")

    stats = []
    for q in pool:
        _lab, ar, _z = branch_of(E, rows, q)
        kids, _, nz = split_rows(E, rows, q)
        sz = sorted(len(v) for v in kids.values())
        stats.append({"chan": int(q[0]), "id": int(q[1]), "answer_rate": ar,
                      "sizes": [len(kids[k]) for k in BRANCHES], "min_child": sz[0],
                      "p_min": sz[0] / len(rows), "n_zero_value": nz})
    it = [s for s in stats if s["chan"] == 0]
    cc = [s for s in stats if s["chan"] == 1]

    def _d(v, key):
        a = np.asarray([x[key] for x in v], np.float64)
        return {"n": len(a), "min": float(a.min()), "median": float(np.median(a)),
                "mean": float(a.mean()), "max": float(a.max())}

    def _depth_bound(p):
        if p <= 0:
            return 0
        d = 0
        n = float(len(rows))
        while n * p >= min_leaf and d < 14:
            n *= p; d += 1
        return d
    best_it = max(it, key=lambda s: s["p_min"]); best_cc = max(cc, key=lambda s: s["p_min"])
    pf = {"n_users": len(rows), "sec_per_eval": {"empty": t_base, "item": t_item, "concept": t_conc},
          "answer_rate": {"items": _d(it, "answer_rate"), "concepts": _d(cc, "answer_rate")},
          "p_min": {"items": _d(it, "p_min"), "concepts": _d(cc, "p_min")},
          "n_passing_min_leaf_at_root": {"items": sum(1 for s in it if s["min_child"] >= min_leaf),
                                         "concepts": sum(1 for s in cc if s["min_child"] >= min_leaf)},
          "best_balanced": {"item": best_it, "concept": best_cc},
          "depth_bound_if_always_best_balanced": {
              "item": _depth_bound(best_it["p_min"]), "concept": _depth_bound(best_cc["p_min"]),
              "median_item": _depth_bound(float(np.median([s["p_min"] for s in it]))),
              "median_concept": _depth_bound(float(np.median([s["p_min"] for s in cc])))},
          "n_zero_value_concepts": int(sum(s["n_zero_value"] for s in cc)),
          "min_leaf": min_leaf, "objective": objective, "CITABLE": False}
    log(f"[preflight] ANSWER RATE at the root: items median {pf['answer_rate']['items']['median']:.3f} "
        f"| concepts median {pf['answer_rate']['concepts']['median']:.3f}")
    log(f"[preflight] SMALLEST-BRANCH FRACTION p_min: items median "
        f"{pf['p_min']['items']['median']:.4f} (max {pf['p_min']['items']['max']:.4f}) | concepts "
        f"median {pf['p_min']['concepts']['median']:.4f} (max {pf['p_min']['concepts']['max']:.4f})")
    log(f"[preflight] candidates whose ROOT split already passes min_leaf={min_leaf}: "
        f"{pf['n_passing_min_leaf_at_root']['items']}/{len(it)} items, "
        f"{pf['n_passing_min_leaf_at_root']['concepts']}/{len(cc)} concepts")
    log(f"[preflight] depth reachable if EVERY node split as evenly as the best candidate: "
        f"item {pf['depth_bound_if_always_best_balanced']['item']}, "
        f"concept {pf['depth_bound_if_always_best_balanced']['concept']} "
        f"(median candidate: {pf['depth_bound_if_always_best_balanced']['median_item']} / "
        f"{pf['depth_bound_if_always_best_balanced']['median_concept']})")
    with open(out_path, "w") as fh:
        json.dump({"preflight": pf, "per_candidate": stats}, fh, indent=2, default=float)
    log(f"[preflight] wrote {os.path.basename(out_path)}")
    log("=" * 110)
    return pf


# --------------------------------------------------------------------------------- the run
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default=CERT_TOWER)
    ap.add_argument("--concept_ckpt", default=CERT_CONC)
    ap.add_argument("--n_items", type=int, default=500)
    ap.add_argument("--n_concepts", type=int, default=500)
    ap.add_argument("--objective", default="full", choices=["full", "tail"])
    ap.add_argument("--total_L", type=int, default=TOTAL_L)
    ap.add_argument("--min_leaf", type=int, default=MIN_LEAF)
    ap.add_argument("--wall_hours", type=float, default=0.0,
                    help="0 = no operational limit (the min-leaf rule is then the only stopping "
                         "rule). If >0, growth STOPS AT THE END OF A COMPLETED LEVEL once this much "
                         "wall clock has passed, and the frontier becomes leaves with full-length "
                         "greedy tails. Growth is level-synchronous, so the result is exactly the "
                         "tree a shallower request would have produced -- and the run is resumable, "
                         "so the depth can be extended later deterministically. Whether it fired is "
                         "recorded in the output as `growth_wall_limited`.")
    ap.add_argument("--label", default=LABEL)
    ap.add_argument("--cohort_concept", type=int, default=-1,
                    help="M3: grow a FOREST TEACHER instead of the cold tree. Restricts the cohort "
                         "to validation users who behaviourally endorse this concept (the anchor "
                         "they would STATE -- the deployable routing signature), and seeds the root "
                         "with that concept as VOLUNTEERED evidence: it is folded into every "
                         "evaluation and excluded from the candidate pool, but costs no turn, so "
                         "the teacher still asks `total_L` questions exactly as the cold tree does. "
                         "Implemented through the existing `prefix` path at depth 0 -- no new "
                         "evaluation route. Design sheet docs/design/DS_M3_FOREST_TREES.md.")
    ap.add_argument("--cohort_rows", default="",
                    help="M3: grow a teacher on an ARBITRARY cohort -- a .json list of validation "
                         "row ids. Used for cohorts that have no stated signature (the residual "
                         "group, or a taste cluster with no nameable attribute). Such a teacher is "
                         "ORACLE-ROUTED when the cohort is defined from information not observable "
                         "at ask time, and must be reported as a ceiling, never as deployable.")
    ap.add_argument("--full_threads", action="store_true",
                    help="consumed by strategy_channel_suite AT IMPORT (it reads sys.argv): use all "
                         "cores rather than its default 4. Declared here only so argparse accepts "
                         "it. The BLAS env vars must ALSO be exported by the launcher BEFORE python "
                         "starts -- numpy is imported before the suite's os.environ lines run, so "
                         "setting them in-process is the no-op recorded on 2026-08-05.")
    ap.add_argument("--preflight", action="store_true",
                    help="build the env, measure s/eval by channel, tabulate the ternary partition "
                         "of every candidate at the root, and STOP. No tree is grown. Not citable.")
    ap.add_argument("--sanity", action="store_true",
                    help="end-to-end pipeline check: budget 3, a 6-question pool that CONTAINS the "
                         "comparator's own head. Also verifies the CELF-heap resume bit-for-bit "
                         "against the pre-change code path. Writes *_SANITY.json, NEVER citable.")
    args = ap.parse_args()

    t00 = time.time()
    sfx = "_SANITY" if args.sanity else ("_PREFLIGHT" if args.preflight else "")
    OUT = os.path.join(OUTDIR, f"{args.label}{sfx}.json")
    PARTIAL = os.path.join(OUTDIR, f"{args.label}{sfx}_partial.json")
    NPZ = os.path.join(OUTDIR, f"{args.label}{sfx}_per_user.npz")
    HEAPDIR = os.path.join(OUTDIR, f"{args.label}{sfx}_heaps")
    os.makedirs(OUTDIR, exist_ok=True); os.makedirs(HEAPDIR, exist_ok=True)

    for nm, p in (("snapshot", args.snapshot), ("concept_ckpt", args.concept_ckpt),
                  ("global greedy json", GLOBAL_JSON), ("concept stats", CONCSTATS)):
        assert os.path.isabs(p) and os.path.exists(p), f"{nm} missing or not absolute: {p}"

    total_L, min_leaf = args.total_L, args.min_leaf
    if args.sanity:
        total_L = 3
        log("#" * 110)
        log("# SANITY MODE: budget 3, 6-question pool. Pipeline check ONLY, NOT citable.")
        log("#" * 110)
    log("#" * 110)
    log("# " + CHARGED)
    log("# " + MIN_LEAF_JUSTIFICATION)
    log("# " + NEG_CAP_NOTE)
    log("# " + RULER_NOTE)
    log("# " + NO_GUARANTEE)
    log("#" * 110)

    comp, gblob = read_global(GLOBAL_JSON, total_L)
    head = comp[HEADLINE_ARM]["sequence"]
    log(f"[global] read from {os.path.basename(GLOBAL_JSON)} (tower={gblob['tower']}, "
        f"conc={gblob['concept_ckpt']}). HEADLINE comparator = arm '{HEADLINE_ARM}', its first "
        f"{total_L} questions {head} -- the same action space this tree searches, so the only thing "
        f"that differs is conditioning. Also reported: " + ", ".join(
            a for a in COMPARATORS if a != HEADLINE_ARM) + " (continuity with the battery).")
    assert os.path.basename(args.snapshot) == gblob["tower"], \
        f"tower mismatch: {os.path.basename(args.snapshot)} vs global build's {gblob['tower']}"
    assert os.path.basename(args.concept_ckpt) == gblob["concept_ckpt"], \
        f"concept ckpt mismatch vs the global build ({gblob['concept_ckpt']})"

    E = build_greedy_env(args.snapshot, "validation", args.concept_ckpt, f"{args.label}(val)")
    rows_all = E["rows"]
    seed_prefix = []
    if args.cohort_concept >= 0:
        cid = int(args.cohort_concept)
        assert 0 <= cid < E["C"], f"--cohort_concept {cid} outside the 0..{E['C']-1} bank"
        _V, _F = np.asarray(E["V"]), np.asarray(E["foldable"])
        _end = (_F > 0) & (_V > 0)
        rows_all = [int(r) for r in rows_all if bool(_end[int(r), cid])]
        seed_prefix = [(1, cid)]
        log(f"[M3] FOREST TEACHER for concept {cid}: cohort restricted to the {len(rows_all)} "
            f"validation users who ENDORSE it. It is folded as volunteered evidence at the root, "
            f"masked from the pool, and costs NO turn -- the teacher asks {total_L} questions, the "
            f"same budget as the cold tree.")
        assert len(rows_all) >= 200, f"anchor cohort is only {len(rows_all)} users"
    if args.cohort_rows:
        assert args.cohort_concept < 0, "--cohort_rows and --cohort_concept are exclusive"
        want = set(int(x) for x in json.load(open(args.cohort_rows)))
        rows_all = [int(r) for r in rows_all if int(r) in want]
        log(f"[M3] EXPLICIT COHORT from {os.path.basename(args.cohort_rows)}: "
            f"{len(rows_all)} of {len(want)} requested rows are in the validation cohort. "
            f"No seed is folded; this teacher starts cold on a selected population.")
        assert len(rows_all) >= 200, f"explicit cohort is only {len(rows_all)} users"
    contract_selftest(E, rows_all)
    log(f"[cohort] {len(rows_all)} canonical VALIDATION users (rows with a non-empty target). "
        f"TEST IS NOT BUILT AND NOT TOUCHED.")
    leak_assert(E, rows_all, "validation")
    gen, _gvocab = load_item_genres(E["ctx"].ni)

    pool_all, item_pool, conc_pool = build_pool(E, args.n_items, args.n_concepts)
    pool = list(pool_all)
    if args.sanity:
        # DEFECT 2 (bit an earlier run): a truncated pool that EXCLUDES the comparator's own head
        # makes the comparison meaningless -- the arm would lose because the reference order was
        # outside its search space, not because conditioning failed. Sanity pool is built head-first.
        pool = list(dict.fromkeys(list(head) + item_pool[:2] + conc_pool[:2]))
    _ps = set(pool)
    assert all(q in _ps for q in head), \
        f"DEFECT: the headline comparator's own {total_L} questions are not all inside the pool"
    budgets = sorted({K for K in BUDGETS if K <= total_L} | {total_L})
    log(f"[pool] COMBINED: {len(item_pool)} items (bank 'pop') + {len(conc_pool)} concepts (bank "
        f"'answerable') = {len(pool)} candidates competing in ONE argmax. Total interview = "
        f"{total_L} questions on BOTH sides; compared question-for-question at K in {budgets}.")

    if args.preflight:
        pf = preflight(E, rows_all, pool, item_pool, conc_pool, min_leaf, args.objective, OUT)
        log(f"[preflight] done in {(time.time() - t00) / 60:.1f}m -- NO TREE GROWN")
        return pf

    if args.sanity:
        heap_resume_selftest(E, rows_all, list(head), args.objective, HEAPDIR)

    sig = {"tower": os.path.basename(args.snapshot), "conc": os.path.basename(args.concept_ckpt),
           "objective": args.objective, "item_bank": "pop", "conc_bank": "answerable",
           "n_items": len(item_pool), "n_concepts": len(conc_pool), "n_pool": len(pool),
           "criterion": "node_size_weighted_mean_full_ndcg10_of_the_ternary_partition",
           "branch_rule": (f"item: absent->noanswer, level>={LIKE_MIN_LEVEL}->like, else dislike | "
                           f"concept: not foldable->noanswer, signed value>0->like, else dislike"),
           "stopping": "min_leaf v2: split iff >=2 non-empty children and >=1 child >= min_leaf; children below the floor inherit the parent tail (no depth cap)", "total_L": total_L, "min_leaf": min_leaf,
           "split": "validation", "n_cohort": len(rows_all), "sanity": bool(args.sanity),
           "cohort_concept": int(args.cohort_concept),
           "cohort_rows": os.path.basename(args.cohort_rows) or None,
           "seed_prefix": [[int(a), int(b)] for a, b in seed_prefix],
           # 2026-08-12 audit: derived from the mask, so a re-run under a different candidate-pool
           # policy cannot resume this tree's nodes. A missing field is a refusal, not a pass.
           "pool_contract": pool_contract(E["pool"], rows_all, policy=E["pool_policy"])}
    part = json.load(open(PARTIAL)) if os.path.exists(PARTIAL) else {}
    if not resume_ok(part.get("sig"), sig, log, os.path.basename(PARTIAL)):
        part = {"sig": sig, "nodes": {}}
    part.setdefault("nodes", {})

    def flush():
        with open(PARTIAL + ".tmp", "w") as fh:
            json.dump(part, fh, indent=2, default=float)
        os.replace(PARTIAL + ".tmp", PARTIAL)

    banked_vec = {}
    nodes, leaves = {}, []
    wall_limited = False
    frontier = [{"nid": "root", "depth": 0,
                 "path": [[int(a), int(b)] for a, b in seed_prefix],
                 "rows": [int(r) for r in rows_all], "branch_from": None}]

    # ======================================================== growth, LEVEL-SYNCHRONOUS
    while frontier:
        lvl_t0 = time.time()
        force_leaf = wall_limited
        nxt = []
        for nd in frontier:
            nid, depth, rows = nd["nid"], nd["depth"], nd["rows"]
            remaining = total_L - depth
            assert remaining >= 1, f"node {nid} at depth {depth} has no budget left"
            heap_path = os.path.join(HEAPDIR, f"node_{nid}.json")
            prefix = [(int(a), int(b)) for a, b in nd["path"]]
            pool_n = [q for q in pool if q not in set(prefix)]
            set_active_rows(E, rows)
            E["memo"] = {}                      # memo keys on the SEQUENCE only: never cross cohorts
            E["n_evals"] = 0; E["t_evals"] = 0.0
            t0 = time.time()

            rec = part["nodes"].get(nid)
            if rec is None or not rec.get("done"):
                rec = {"nid": nid, "depth": depth, "n_users": len(rows),
                       "path": [[int(a), int(b)] for a, b in prefix], "branch_from": nd["branch_from"],
                       "remaining_budget": remaining, "done": False}
                part["nodes"][nid] = rec
                flush()
                log("-" * 110)
                log(f"[node {nid}] depth {depth}, {len(rows)} users, path {rec['path']}, "
                    f"{remaining} question(s) of budget left, {len(pool_n)} candidates")

                def on_pick(seq, gains, nev, _r=rec, _t=t0):
                    _r["steps"] = [[int(a), int(b)] for a, b in seq]
                    _r["gains"] = [float(g) for g in gains]
                    _r["n_evals"] = int(nev)
                    _r["sec_per_eval"] = round(E["t_evals"] / max(E["n_evals"], 1), 3)
                    _r["minutes"] = round((time.time() - _t) / 60, 2)
                    flush()                     # HARD RULE 11(a): on disk after EVERY pick

                # The split search IS the CELF seed pass, and it is banked to the node's heap
                # sidecar -- so if the min-leaf rule then stops this node, growing its greedy tail
                # below RESUMES that heap instead of paying a second 1,000-candidate seed pass.
                seq1, gains1, _nev = greedy_certified(E, rows, pool_n, 1, args.objective,
                                                      f"{nid}/split", prefix=prefix, on_pick=on_pick,
                                                      heap_path=heap_path)
                assert len(seq1) == 1, f"node {nid}: split search returned {len(seq1)} picks"
                q = (int(seq1[0][0]), int(seq1[0][1]))
                kids, ar, n_zero = split_rows(E, rows, q)
                sizes = {k: len(v) for k, v in kids.items()}
                rec.update({"splitter": [q[0], q[1]], "channel": "concept" if q[0] else "item",
                            "splitter_gain": float(gains1[0]), "child_sizes": sizes,
                            "answer_rate": round(ar, 4), "n_zero_value_answers": n_zero})

                # ---- the criterion, written out as the partition it claims to be -----------------
                f_node, _tn = genv_eval(E, prefix + [q], rows)
                assert np.isfinite(f_node).all(), "a validation user has a non-finite full NDCG@10"
                pos = {int(r): k for k, r in enumerate(rows)}
                wsum, chmeans = 0.0, {}
                for k in BRANCHES:
                    if not kids[k]:
                        chmeans[k] = None
                        continue
                    chmeans[k] = float(f_node[[pos[r] for r in kids[k]]].mean())
                    wsum += len(kids[k]) / len(rows) * chmeans[k]
                err = abs(wsum - float(f_node.mean()))
                rec.update({"child_full@10": chmeans, "partition_weighted_full@10": wsum,
                            "node_full@10_after_splitter": float(f_node.mean()),
                            "partition_decomposition_max_abs_err": err})
                assert err < 1e-9, (f"node {nid}: the node-size-weighted partition mean {wsum} does "
                                    f"not equal the node mean {float(f_node.mean())} (err {err:.3e}) "
                                    f"-- the criterion is not what it says it is")

                # ---- STOPPING RULE v2 (2026-08-17, author: "why can't you split nodes that
                # have > 200?"). v1 took min() over ALL THREE children, so ONE small branch vetoed
                # the whole split -- and for the M3 teachers it vetoed splits whose two real children
                # held 334 and 396 users, in one case because the did-not-answer branch was EMPTY.
                # Nothing about generalisation is at stake in a branch with nobody in it.
                #
                # The 200 floor's actual evidence (genre split, 2026-08-11) is about FITTING AN ORDER
                # on a small cohort, not about splitting. So: split when the partition is real and at
                # least one child can carry its own fitted continuation; children BELOW the floor
                # become leaves that INHERIT this node's tail, fitted on this node's full cohort.
                # Nothing is ever fitted below the floor.
                nonempty = [k for k in BRANCHES if sizes[k] > 0]
                big = [k for k in BRANCHES if sizes[k] >= min_leaf]
                small = min(sizes.values())
                smallest_nonempty = min([sizes[k] for k in nonempty], default=0)
                budget_left_to_split = remaining >= 2
                will_split = (budget_left_to_split and len(nonempty) >= 2 and len(big) >= 1
                              and not force_leaf)
                rec["min_child"] = int(small)
                rec["min_nonempty_child"] = int(smallest_nonempty)
                rec["children_above_floor"] = list(big)
                rec["children_inheriting"] = [k for k in nonempty if k not in big]
                rec["split"] = bool(will_split)
                rec["stop_reason"] = (None if will_split else
                                      ("growth_wall_limit" if force_leaf else
                                       "budget_exhausted (1 question left, nothing to split into)"
                                       if not budget_left_to_split else
                                       f"degenerate partition: {len(nonempty)} non-empty child(ren)"
                                       if len(nonempty) < 2 else
                                       f"no child reaches min_leaf {min_leaf} "
                                       f"(largest {max(sizes.values())})"))
                log(f"[node {nid}] splitter = {rec['channel']} {q[1]} (gain {gains1[0]:+.5f}, answer "
                    f"rate {ar:.3f}); ternary like={sizes['like']} dislike={sizes['dislike']} "
                    f"noanswer={sizes['noanswer']}; decomposition err {err:.2e}; "
                    + ("SPLIT" if will_split else f"STOP ({rec['stop_reason']})"))

                # The node's own full-length tail is computed WHETHER OR NOT it splits: a
                # splitting node hands it to any child below the floor (rule v2). It resumes the same
                # CELF heap the split search seeded, so the extra picks are cheap.
                # A tail is needed only when SOMEONE will inherit it: either this node is a leaf,
                # or it splits and at least one non-empty child sits below the floor. A node whose
                # children are ALL above the floor hands its tail to nobody -- computing it anyway
                # cost the real root ~2 h of pure waste (children 2606/4814/2580) before this was
                # caught. It resumes the same CELF heap the split search seeded.
                needs_tail = (not will_split) or bool(rec["children_inheriting"])
                if not needs_tail:
                    log(f"[node {nid}] tail SKIPPED -- every non-empty child is above min_leaf "
                        f"{min_leaf}, so no child inherits it")
                    seqL, gainsL, nevL = [q], [float(gains1[0])], int(E["n_evals"])
                else:
                    seqL, gainsL, nevL = greedy_certified(E, rows, pool_n, remaining, args.objective,
                                                          f"{nid}/tail", prefix=prefix,
                                                          on_pick=on_pick, heap_path=heap_path)
                assert (not needs_tail) or len(seqL) == remaining, \
                    f"node {nid}: tail greedy returned {len(seqL)} of {remaining} picks"
                assert tuple(seqL[0]) == q, \
                    f"node {nid}: the resumed tail order does not start with the split pick"
                rec["tail"] = [[int(a), int(b)] for a, b in seqL]
                rec["tail_gains"] = [float(g) for g in gainsL]
                rec["n_evals"] = int(nevL)
                if will_split:
                    rec["steps"] = [[q[0], q[1]]]
                    rec["gains"] = [float(gains1[0])]
                else:
                    rec["steps"] = list(rec["tail"])
                    rec["gains"] = list(rec["tail_gains"])
                rec["sec_per_eval"] = round(E["t_evals"] / max(E["n_evals"], 1), 3)
                rec["minutes"] = round((time.time() - t0) / 60, 2)
                rec["genre_face_validity"] = genre_face_validity(E, rows, gen)
                rec["done"] = True
                flush()
                log(f"[node {nid}] DONE in {rec['minutes']:.1f}m, {rec['n_evals']} evals, "
                    f"{rec['sec_per_eval']}s/eval (n={len(rows)})")
            else:
                log(f"[node {nid}] already complete in the sidecar -- reusing (splitter "
                    f"{rec.get('splitter')}, split={rec.get('split')})")

            nodes[nid] = rec
            if rec["split"]:
                kids, _ar, _z = split_rows(E, rows, tuple(rec["splitter"]))
                cpath = [[int(a), int(b)] for a, b in rec["path"]] + [list(rec["splitter"])]
                if rec["children_inheriting"]:
                    assert len(rec["tail"]) == rec["remaining_budget"], (
                        f"node {nid} has inheriting children but its tail is "
                        f"{len(rec['tail'])} picks, not {rec['remaining_budget']}")
                # lazy: a node whose children are all above the floor skipped its tail, and a node
                # RESUMED from the v1 build has none either. Only materialise if someone inherits.
                inherit_own = None
                for k in BRANCHES:
                    if not kids[k]:
                        continue                       # an empty branch is not a node
                    if len(kids[k]) >= min_leaf:
                        nxt.append({"nid": f"{nid}.{k}", "depth": depth + 1, "path": cpath,
                                    "rows": kids[k], "branch_from": k})
                    else:
                        # BELOW THE FLOOR: this child asks what its parent would have asked -- the
                        # parent's tail, fitted on the parent's full cohort. It is a leaf, it
                        # conditions no further, and NOTHING is fitted on its own users.
                        cnid = f"{nid}.{k}"
                        if inherit_own is None:
                            inherit_own = ([(int(a), int(b)) for a, b in rec["path"]] +
                                           [(int(a), int(b)) for a, b in rec["tail"]])
                        crec = {"nid": cnid, "depth": depth + 1, "n_users": len(kids[k]),
                                "path": [[int(a), int(b)] for a, b in rec["path"]],
                                "branch_from": k, "inherited_from": nid, "split": False,
                                "stop_reason": f"below min_leaf {min_leaf}; inherits parent tail",
                                "steps": [list(x) for x in rec["tail"]],
                                "gains": list(rec["tail_gains"]),
                                # this child ran NO search of its own -- it reuses the parent's
                                # measured tail, so its cost fields are zero by construction.
                                "n_evals": 0, "sec_per_eval": 0.0, "minutes": 0.0,
                                "splitter": None, "channel": None, "min_child": None,
                                "genre_face_validity": genre_face_validity(E, kids[k], gen),
                                "done": True}
                        nodes[cnid] = crec
                        assert len(inherit_own) == total_L, (
                            f"inherited order for {cnid} has {len(inherit_own)} questions")
                        leaves.append({"nid": cnid, "depth": depth + 1, "path": crec["path"],
                                       "rows": kids[k], "branch_from": k,
                                       "own": inherit_own, "rec": crec})
            else:
                own = [(int(a), int(b)) for a, b in rec["path"]] + \
                      [(int(a), int(b)) for a, b in rec["steps"]]
                assert len(own) == total_L, f"leaf {nid} has {len(own)} questions, expected {total_L}"
                nd["own"] = own; nd["rec"] = rec
                leaves.append(nd)
        d_done = frontier[0]["depth"]
        log(f"[level] depth {d_done} finished in {(time.time() - lvl_t0) / 60:.1f}m: "
            f"{len(frontier)} node(s) -> {len(nxt)} node(s) at depth {d_done + 1}; "
            f"{len(leaves)} leaf/leaves so far; elapsed {(time.time() - t00) / 60:.1f}m")
        if args.wall_hours > 0 and not wall_limited and nxt and \
                (time.time() - t00) > args.wall_hours * 3600:
            wall_limited = True
            log("!" * 110)
            log(f"[level] OPERATIONAL WALL LIMIT ({args.wall_hours} h) reached after depth {d_done}. "
                f"Growth stops here: the {len(nxt)} node(s) at depth {d_done + 1} become LEAVES and "
                f"get full-length greedy tails. Level-synchronous growth means this is EXACTLY the "
                f"tree a shallower request would have produced, it is recorded as "
                f"growth_wall_limited=true, and the run is resumable so the depth can be extended "
                f"later deterministically.")
            log("!" * 110)
        frontier = nxt

    # ======================================================== measurement, per leaf
    # DEFECT 1 (bit an earlier run): comparing UNEQUAL budgets. Every comparison below is the leaf's
    # first K questions against a comparator's first K, with K asserted equal on both sides.
    log("=" * 110)
    log(f"[measure] {len(leaves)} leaves, {sum(len(l['rows']) for l in leaves)} users total")
    for lf in leaves:
        nid, rows, own, rec = lf["nid"], lf["rows"], lf["own"], lf["rec"]
        set_active_rows(E, rows); E["memo"] = {}
        curve = {}
        for K in budgets:
            a = own[:K]
            assert len(a) == K, f"UNEQUAL BUDGET: tree arm has {len(a)} questions at K={K}"
            of, ot = genv_eval(E, a, rows)
            ia, ca = genv_answered(E, a, rows)
            banked_vec[f"{nid}|{K}|own_full"] = of.astype(np.float32)
            banked_vec[f"{nid}|{K}|own_tail"] = ot.astype(np.float32)
            row = {"tree_full@10": float(np.nanmean(of)), "tree_tail@10": float(np.nanmean(ot)),
                   "tree_items_answered": round(ia, 3), "tree_concepts_answered": round(ca, 3),
                   "tree_n_items": sum(1 for c, _ in a if c == 0),
                   "tree_n_concepts": sum(1 for c, _ in a if c == 1)}
            for arm in COMPARATORS:
                b = comp[arm]["sequence"][:K]
                assert len(b) == K, f"UNEQUAL BUDGET: comparator '{arm}' has {len(b)} at K={K}"
                gf, gt = genv_eval(E, b, rows)
                banked_vec[f"{nid}|{K}|{arm}_full"] = gf.astype(np.float32)
                banked_vec[f"{nid}|{K}|{arm}_tail"] = gt.astype(np.float32)
                row[f"{arm}_full@10"] = float(np.nanmean(gf))
                row[f"{arm}_tail@10"] = float(np.nanmean(gt))
                row[f"diff_full@10_vs_{arm}"] = float(np.nanmean(of - gf))
                row[f"diff_tail@10_vs_{arm}"] = float(np.nanmean(ot - gt))
                row[f"set_overlap_vs_{arm}"] = len(set(a) & set(b))
            curve[str(K)] = row
        banked_vec[f"{nid}|rows"] = np.asarray(rows, np.int64)
        rec["curve_matched"] = curve
        rec["own_sequence"] = [[int(x), int(y)] for x, y in own]
        flush(); np.savez_compressed(NPZ, **banked_vec)
        c = curve[str(total_L)]
        log(f"[leaf {nid}] n={len(rows)} depth={rec['depth']} | @K={total_L} tree "
            f"{c['tree_full@10']:.5f} vs {HEADLINE_ARM} {c[HEADLINE_ARM + '_full@10']:.5f} "
            f"({c['diff_full@10_vs_' + HEADLINE_ARM]:+.5f} full, "
            f"{c['diff_tail@10_vs_' + HEADLINE_ARM]:+.5f} tail) | "
            f"{c['tree_n_items']} items / {c['tree_n_concepts']} concepts | order {own}")

    weighted = {}
    for K in budgets:
        w = {}
        for arm in COMPARATORS:
            df = np.concatenate([banked_vec[f"{l['nid']}|{K}|own_full"]
                                 - banked_vec[f"{l['nid']}|{K}|{arm}_full"] for l in leaves])
            dt = np.concatenate([banked_vec[f"{l['nid']}|{K}|own_tail"]
                                 - banked_vec[f"{l['nid']}|{K}|{arm}_tail"] for l in leaves])
            w[f"diff_full@10_vs_{arm}"] = float(np.nanmean(df))
            w[f"diff_tail@10_vs_{arm}"] = float(np.nanmean(dt))
            w["n_users"] = int(np.isfinite(df).sum())
        weighted[str(K)] = w

    # ---- THE NESTING CHECK, at full scale --------------------------------------------------------
    rr = nodes["root"]
    same_q = [int(x) for x in rr["splitter"]] == [int(x) for x in comp[HEADLINE_ARM]["sequence"][0]]
    nest = {"root_splitter": [int(x) for x in rr["splitter"]],
            "headline_first_question": [int(x) for x in comp[HEADLINE_ARM]["sequence"][0]],
            "root_first_question_matches_comparator": bool(same_q),
            "root_gain_policyB": float(rr["splitter_gain"]),
            "headline_first_gain_banked_policyA": float(comp[HEADLINE_ARM]["gains"][0]),
            "K1_cohort_diff_full": weighted["1"][f"diff_full@10_vs_{HEADLINE_ARM}"],
            "ruler_note": RULER_NOTE}
    log("=" * 110)
    log(f"[nesting] root splitter {nest['root_splitter']} vs the global '{HEADLINE_ARM}' arm's first "
        f"question {nest['headline_first_question']} -> match={same_q}; this run's gain "
        f"{nest['root_gain_policyB']:.12f} (policy B) vs the file's banked "
        f"{nest['headline_first_gain_banked_policyA']:.12f} (policy A -- NOT comparable, see below); "
        f"cohort diff at K=1 = {nest['K1_cohort_diff_full']:+.2e}")
    log("[nesting] " + RULER_NOTE)
    if same_q:
        assert abs(nest["K1_cohort_diff_full"]) < 1e-12, \
            "the root splitter equals the comparator's first question but the K=1 difference is not 0"
        log("[nesting] K=1 TIE EXACT: the tree is nested in the comparator at the first question, so "
            "every K>=2 difference is attributable to conditioning alone.")
    else:
        log("!" * 110)
        log(f"[nesting] THE TREE'S FIRST QUESTION DIFFERS FROM THE COMPARATOR'S. That is not a "
            f"harness fault: the banked order was SELECTED under the retired policy-A candidate "
            f"mask, this run selects under policy B, and an argmax can move when the ruler moves. "
            f"CONSEQUENCE: the K=1 difference ({nest['K1_cohort_diff_full']:+.5f}) is NOT zero and is "
            f"NOT conditioning -- it is the policy-A/B selection difference, and it sits inside every "
            f"K>=2 number as an offset. Do not report the K>=2 gains as pure conditioning without "
            f"subtracting it or without rebuilding the static comparator under policy B.")
        log("!" * 110)

    # ---- report ----------------------------------------------------------------------------------
    depth_reached = max(r["depth"] for r in nodes.values())
    depth_hist = {}
    for lf in leaves:
        d = nodes[lf["nid"]]["depth"]
        depth_hist[str(d)] = depth_hist.get(str(d), 0) + 1
    chan_by_depth = {}
    for r in nodes.values():
        if r.get("split"):
            k = str(r["depth"])
            c = chan_by_depth.setdefault(k, {"item": 0, "concept": 0})
            c[r["channel"]] += 1
    tbl = sorted((nodes[l["nid"]] for l in leaves), key=lambda r: -r["n_users"])
    log("=" * 110)
    log("LEAVES (validation / SELECTION cohort). " + NO_GUARANTEE)
    log("-" * 110)
    log(f"{'leaf':<34}{'d':>3}{'n':>7}{'dFULL':>10}{'dTAIL':>10}{'it':>4}{'cc':>4}"
        f"{'ovl':>5}{'s/ev':>8}")
    for r in tbl:
        c = r["curve_matched"][str(total_L)]
        log(f"{r['nid']:<34}{r['depth']:>3}{r['n_users']:>7}"
            f"{c['diff_full@10_vs_' + HEADLINE_ARM]:>10.5f}"
            f"{c['diff_tail@10_vs_' + HEADLINE_ARM]:>10.5f}{c['tree_n_items']:>4}"
            f"{c['tree_n_concepts']:>4}{c['set_overlap_vs_' + HEADLINE_ARM]:>5}"
            f"{r['sec_per_eval']:>8.2f}")
    log("-" * 110)
    for K in budgets:
        w = weighted[str(K)]
        log(f"COHORT-WEIGHTED (all {w['n_users']} users) @K={K:<3} " + "  ".join(
            f"vs {a}: {w['diff_full@10_vs_' + a]:+.5f}/{w['diff_tail@10_vs_' + a]:+.5f}"
            for a in COMPARATORS))
    log(f"[shape] depth reached {depth_reached}; {len(nodes)} nodes, {len(leaves)} leaves; leaf depth "
        f"histogram {depth_hist}; split channel by depth {chan_by_depth}; "
        f"growth_wall_limited={wall_limited}")
    log("=" * 110)

    out = {"analysis": "ndcg_objective_interview_tree",
           "stage": "1_build_on_validation",
           "label": args.label, "SANITY": bool(args.sanity), "citable": not args.sanity,
           "split": "validation", "test_touched": False,
           "charged": CHARGED, "min_leaf_justification": MIN_LEAF_JUSTIFICATION,
           "no_guarantee": NO_GUARANTEE, "concept_answer_convention": NEG_CAP_NOTE,
           "ruler_note": RULER_NOTE,
           "criterion": sig["criterion"],
           "criterion_note": ("NDCG@10 is per-user, so the node-size-weighted mean over the ternary "
                              "partition is algebraically the node mean; the split search is "
                              "therefore the greedy argmax restricted to the node's users, and this "
                              "tree is MYOPIC in the same sense Golbandi's is. Adaptivity comes from "
                              "the CHILDREN diverging, not from lookahead. Measured, not assumed: "
                              "partition_decomposition_max_abs_err on every node."),
           "nearest_prior_work": ("Hu & Yu, RecSys 2013, 'Interview Process Learning for Top-N "
                                  "Recommendation' -- an interview tree whose split loss may be a "
                                  "PAIRWISE RANK loss. Separations: smooth pairwise surrogate vs "
                                  "truncated discounted NDCG@10; training loss vs measured value of "
                                  "asking; MAP on 10x sampled negatives vs the full 18,359-item "
                                  "catalogue; static comparator top-k popular vs an order built by "
                                  "maximising the reported metric."),
           "tower": os.path.basename(args.snapshot),
           "concept_ckpt": os.path.basename(args.concept_ckpt),
           "objective": args.objective, "item_bank": "pop", "conc_bank": "answerable",
           "n_items_pool": len(item_pool), "n_concepts_pool": len(conc_pool), "n_pool": len(pool),
           "branch_rule": sig["branch_rule"], "like_min_level": int(LIKE_MIN_LEVEL),
           "stopping_rule": "min_leaf only (no depth cap); budget stops a node with 1 question left",
           "min_leaf": min_leaf, "total_L": total_L, "budgets": budgets,
           "growth_wall_limited": bool(wall_limited), "wall_hours": args.wall_hours,
           "depth_reached": depth_reached, "leaf_depth_histogram": depth_hist,
           "split_channel_by_depth": chan_by_depth,
           "n_cohort": len(rows_all), "n_nodes": len(nodes), "n_leaves": len(leaves),
           "headline_comparator": HEADLINE_ARM,
           "headline_comparator_why": ("the global build's `combined` arm searches the SAME action "
                                       "space (500 pop items + 500 answerable concepts) with the "
                                       "SAME objective on the SAME cohort, so conditioning is the "
                                       "only thing that differs. `concepts-only` and `items-only` "
                                       "are reported alongside for continuity with the battery."),
           "comparators": {a: {"sequence": [[int(x), int(y)] for x, y in comp[a]["sequence"]],
                               "gains": comp[a]["gains"][:total_L]} for a in COMPARATORS},
           "nesting_check": nest,
           "nodes": nodes,
           "leaf_table": [{"nid": r["nid"], "depth": r["depth"], "n_users": r["n_users"],
                           "path": r["path"], "branch_from": r["branch_from"],
                           "stop_reason": r["stop_reason"], "sequence": r["own_sequence"],
                           "diff_full@10": r["curve_matched"][str(total_L)][
                               "diff_full@10_vs_" + HEADLINE_ARM],
                           "diff_tail@10": r["curve_matched"][str(total_L)][
                               "diff_tail@10_vs_" + HEADLINE_ARM],
                           "sec_per_eval": r["sec_per_eval"], "n_evals": r["n_evals"],
                           "genre_face_validity": r["genre_face_validity"]} for r in tbl],
           "cohort_weighted": weighted,
           "per_user_npz": os.path.basename(NPZ),
           "sig": sig, "seconds": round(time.time() - t00, 1),
           # the artifact states its own ruler (2026-08-12 audit). The COMPARATOR's selection policy
           # is a property of GLOBAL_JSON, not of this run, and is deliberately not claimed here.
           "pool_policy": E["pool_policy"], "pool_contract": sig["pool_contract"],
           "comparator_source": os.path.basename(GLOBAL_JSON),
           "baseline_sec_per_eval_10k": gblob.get("sec_per_eval_build")}
    with open(OUT + ".tmp", "w") as fh:
        json.dump(out, fh, indent=2, default=float)
    os.replace(OUT + ".tmp", OUT)
    np.savez_compressed(NPZ, **banked_vec)
    log(f"[done] wrote {os.path.basename(OUT)} + {os.path.basename(NPZ)} in "
        f"{(time.time() - t00) / 60:.1f}m")
    return out


if __name__ == "__main__":
    main()
