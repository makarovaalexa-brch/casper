r"""P2 -- GENERALISATION ACROSS STATES: the evidence-seeded fine-tune.

Design sheet: `docs/design/PAPER_B_POLICY_DESIGN.md` **PART I.2 is binding**. Committed before any
run (HARD RULE 10). Validation only; the 10,000 canonical TEST users are never built, never loaded,
never named -- there is no `--split` in this file.

WHAT IS BEING ASKED
-------------------
R measured that the distilled policy is INERT on states it did not create (95-99% of users receive
the identical unseeded interview), and R2 measured that the belief-only variant is SENSITIVE but not
INFORMED (it churns 9-13 of 16 questions and the churn costs -0.012). I.2 asks the obvious next
question: if the policy is TAUGHT on evidence-seeded states, does the deviation stop costing?

THE TRAINING MIXTURE (design sheet I.2, declared constant, NOT tuned post hoc)
-----------------------------------------------------------------------------
Each batch is 7/8 REPLAY -- the original E1 tree-state labels, the proven anti-forgetting mechanism
-- and 1/8 SEEDED, from **209 off-manifold states**, not the 138 Part I.2 named. See
`load_offmanifold_extra` for the census by source and why the count was corrected upward.

⚠⚠ TEACHER HONESTY, CARRIED ON EVERY NUMBER, AND IT IS NOT UNIFORM ACROSS SOURCES.
  * the 138 random-prefix states: the label is the measured-NDCG argmax over the **~22 IN-POOL
    candidates of the 50 that sweep sampled** (20 pool + 30 catalogue). NOT the argmax of the
    1,000-pool. This is the weakest teacher in the mixture and it carries the most states.
  * the 42 below-floor states: **exact** -- ~996 of the 1,000 pool candidates were measured.
  * the 30 static-order states: **exact** -- the banked `pick` is greedy's own chosen question.
An imperfect teacher is acceptable for a capability test and must be named. The full-pool upgrade
for the 138 is priced in I.2 at 12-23 h and was not run.

⚠ AND THE LABELS ARE MEASURED ON THE COHORT THEY ARE EVALUATED ON. The sweep measured NDCG on the
10,000 validation users, so this fine-tune fits labels derived from validation NDCG and is then
scored on validation. That is the standing protocol of this whole chapter (the tree itself was grown
this way, and the single unspent test look is reserved), but it is a limitation and is stated, not
smuggled.

★ THE ENCODING QUESTION, AND WHY BOTH FORMS ARE TRAINED
--------------------------------------------------------
A random-prefix state and a volunteered-evidence state are the same BELIEF with different
BOOKKEEPING: in the prefix state the evidence arrived through questions, so the explicit ternary
asked-history is populated; in the evaluation state the user volunteered items, so `z_t` has moved
and the asked-history is EMPTY. R2 measured that this exact pairing -- moved `z`, empty history --
is what E1 never saw and what pins it to its opening. Training only the as-asked form would
therefore teach nothing that the evaluation can see, and the falsifier would fire for a bookkeeping
reason rather than a real one. So every seeded state is presented in BOTH forms, 50/50 inside the
seeded eighth:

  (a) AS ASKED       z = the prefix-folded latent, history = the prefix questions + their ternary
                     outcomes, position t = k.
  (b) AS VOLUNTEERED z = the same latent, history EMPTY, position t = 0 -- the encoding the
                     evaluation actually presents.

The measured label is a property of the evidence, not of how it arrived: the sweep folded the prefix
and measured each candidate's marginal NDCG gain from there, so it is the correct target for both
forms. This is declared here, before the run, and reported with the result.

THE EVALUATION (design sheet I.2)
---------------------------------
A FRESH volunteered draw -- `--draw_seed 424242`, disjoint from R/R2's 98765 -- through R's own
harness functions (`policy_resume.draw_volunteered`, `rollout_seeded`, `build_seqs`), so the
measurement is the one R2 published and not a lookalike.

  spanning gate (HARD RULE 8)  the fine-tuned policy's UNSEEDED K=16 validation rollout must not
                               fall below E1's 0.28355. If it fails, the arm is rejected whatever
                               else it does.
  ★ the deviation-cost flip    fine-tuned policy MINUS ITS OWN FROZEN ORDER on fresh seeded states.
                               R2 measured -0.01246 / -0.01234 / -0.00860 at k = 2/4/8. The bar is
                               >= 0 with the interval excluding a loss.
  vs the tree                  paired CI. The capability sentence needs "not worse", not "beats".
  redundancy                   with an interval this time.
  controls                     the seeded labels shuffled ACROSS states (HARD RULE 5), and the
                               anti-collapse diagnostics.

CHECKPOINT SELECTION, DECLARED BEFORE THE RUN: the LAST epoch whose unseeded K=16 validation rollout
clears the spanning gate. If no epoch clears it, the arm is rejected and reported rejected. No
epoch is chosen by anything the evaluation can see.
"""
import os
import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)

import glob
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn

import policy_net as PN
import policy_resume as PR
from policy_net import (OUTDIR, SEED, TOTAL_L, STATIC_JSON, STATIC_VAL_FULL, STATIC_VAL_TAIL,
                        VAL_M0_FULL, E1_VAL_FULL, E1_PER_USER, Policy, build_phi, paired_bootstrap,
                        pep_divergence_from_arrays, channel_mix, load_traj, make_batch, rollout,
                        teacher_forced_agreement, log)

P2_DIR = os.path.join(OUTDIR, "p2_seeded")
SAMPLE_DIR = os.path.join(OUTDIR, "value_sample_partial")
SEEDED_NPZ = os.path.join(P2_DIR, "p2_seeded_states.npz")
SEEDED_JSON = os.path.join(P2_DIR, "p2_seeded_states.json")
DRAW_SEED = 424242                 # ★ disjoint from R / R2's 98765. Declared, not silent.
SEEDED_FRAC = 1.0 / 8.0            # declared constant (I.2), not tuned post hoc
MAX_K = 15


# ============================================================================ the other off-manifold states
DEEP_DIR = os.path.join(OUTDIR, "policy_labels_deep_partial")
JSONL = os.path.join(OUTDIR, "value_dataset.jsonl")


def load_offmanifold_extra():
    """★ THE STATE COUNT, CORRECTED UPWARD (author's catch, 2026-08-15). Part I.2 said "138
    evidence-seeded states" and that undercounted: the 425-state / 72,717-triple bank was built for
    the VALUE model and every one of its states also carries a measured ACTION label that has never
    trained anything. Counted rather than assumed, by source:

      216  the tree's own greedy-tail states (`ndcgtree16_heaps`, `ndcgtree_heaps`) -- ON E1's
           trajectory manifold. E1 already trains on these; they are the replay, not new material,
           and they are EXCLUDED from the seeded set.
       42  `policy_labels_deep_partial` -- phase 3's below-floor sweeps. FULL-POOL: ~996 of the 1,000
           candidates measured at each state, so the label is the EXACT pool argmax. Rows banked.
      138  the random-prefix sweeps. Label = argmax over the ~22 IN-POOL candidates of the 50 sampled.
       31  `greedy_static_policyB_{items,concepts}_heaps` -- STATIC-ORDER states, cohort = all 10,000
           users. The banked `pick` triple IS greedy's chosen next question, i.e. the exact argmax, so
           the label is exact even though only the heap survivors were banked. ONE of the 31 is the
           EMPTY state, which E1 already handles with the tree's root; relabelling it would disturb
           behaviour the policy already has (the phase-3 failure mode Part I.0 warns about), so it is
           dropped and the drop is counted.

    Returns the two banked-row sources this function owns (deep + static); the random prefixes are
    read by the caller from their own sidecars.
    """
    out = []
    for fp in sorted(glob.glob(os.path.join(DEEP_DIR, "*.json"))):
        if os.path.basename(fp) == "_sig.json":
            continue
        d = json.load(open(fp))
        if not d.get("done"):
            continue
        out.append({"kind": "deep_full_pool", "sid": d["sid"],
                    "prefix": [(int(a), int(b)) for a, b in d["prefix"]],
                    "rows": [int(r) for r in d["rows"]],
                    "gains": {(int(q.split("_")[0]), int(q.split("_")[1])): float(v[0])
                              for q, v in d["vals"].items()},
                    "label_q": None, "teacher": "exact argmax over ~996 of 1,000 pool candidates"})
    st, meta = {}, {}
    for line in open(JSONL):
        d = json.loads(line)
        if not d["source_file"].split("/")[0].startswith("greedy_static"):
            continue
        sid = d["state_id"]
        meta[sid] = {"asked": [(int(a), int(b)) for a, b in d["asked_set"]],
                     "depth": int(d["depth"]), "n_rows": int(d["n_rows"])}
        st.setdefault(sid, {})[(int(d["question"][0]), int(d["question"][1]))] = (
            float(d["gain"]), bool(d["in_pool"]), d["selection_tag"])
    n_empty = 0
    for sid, g in st.items():
        if meta[sid]["depth"] == 0:
            n_empty += 1
            continue                       # E1 already handles the empty state; do not disturb it
        picks = [q for q, (_gg, _ip, t) in g.items() if t == "pick"]
        if not picks:
            continue
        out.append({"kind": "static_order", "sid": sid, "prefix": meta[sid]["asked"],
                    "rows": None,          # cohort = all 10,000 users, filled in by the caller
                    "gains": {q: v[0] for q, v in g.items() if v[1]},
                    "label_q": picks[0],
                    "teacher": "greedy's own chosen next question -- the exact pool argmax"})
    log(f"[extra] {sum(1 for r in out if r['kind'] == 'deep_full_pool')} full-pool below-floor "
        f"states + {sum(1 for r in out if r['kind'] == 'static_order')} static-order states "
        f"({n_empty} empty-state static entries dropped: E1 already handles that state)")
    return out


# ============================================================================ build the seeded set
def build_seeded(args):
    """Turn the 19 random-prefix sweeps into (state, label) training examples, and PRINT THE
    COMPOSITION before any training time is spent."""
    os.makedirs(P2_DIR, exist_ok=True)
    from strategy_channel_suite import (build_greedy_env, contract_selftest, genv_eval, genv_mean,
                                        genv_latent)
    from ndcg_tree import build_pool, leak_assert, branch_of_per_user
    t00 = time.time()

    files = sorted(glob.glob(os.path.join(SAMPLE_DIR, "prefix_*.json")))
    assert files, f"no random-prefix sweeps in {SAMPLE_DIR}"
    extra = load_offmanifold_extra()
    Dva = load_traj(train=False)
    pool = Dva["pool"]
    pool_index = {(int(a), int(b)): i for i, (a, b) in enumerate(pool)}

    E = build_greedy_env(args.snapshot, "validation", args.concept_ckpt, "p2/build")
    contract_selftest(E, E["rows"]); leak_assert(E, E["rows"], "p2/build")
    rows = E["rows"]
    assert len(rows) == 10000
    f0, t0_ = genv_mean(E, [], rows)
    assert round(f0, 8) == VAL_M0_FULL, f"CANONICAL SNAP FAILED: {f0!r}"
    log(f"[gate] canonical m=0 snap full={f0!r} tail={t0_!r} -- exact")
    pool_e, _i, _c = build_pool(E, 500, 500)
    assert pool_e == pool, "the env's pool is not the trajectory bank's pool"
    static_seq = [(int(a), int(b)) for a, b in json.load(open(STATIC_JSON))["sequence"]]
    sfv, _stv = genv_eval(E, static_seq, rows)
    assert abs(float(np.nanmean(sfv)) - STATIC_VAL_FULL) < 1e-12, "BANKED-HEADLINE ASSERT FAILED"
    log(f"[gate] greedy_static_policyB_combined reproduces EXACTLY: {float(np.nanmean(sfv))!r}")
    rowpos = {int(r): j for j, r in enumerate(rows)}

    Zs, HQ, HO, KK, LAB, PFX, SRC, SID = [], [], [], [], [], [], [], []
    prefix_meta, n_states_used, n_cat_argmax = [], 0, 0
    next_sid = 0
    state_meta = []
    for fp in files:
        d = json.load(open(fp))
        assert d["test_touched"] is False and d["sig"]["split"] == "validation"
        assert d["sig"]["n_pool"] == len(pool)
        pi, k = int(d["prefix_index"]), int(d["k"])
        prefix = [(int(a), int(b)) for a, b in d["prefix"]]
        cands = [(int(a), int(b)) for a, b in d["candidates"]]
        inp = [bool(x) for x in d["candidate_in_pool"]]
        assert len(prefix) == k <= MAX_K

        def _label(state):
            """★ the measured argmax, restricted to the IN-POOL candidates -- the policy's action
            space is the 1,000-pool, so a catalogue argmax is not expressible. Both are counted."""
            g = state["gains"]
            v = [g[f"{q[0]}_{q[1]}"] for q in cands]
            best_any = int(np.argmax(v))
            vp = [(v[j], j) for j in range(len(cands)) if inp[j]]
            best_pool = max(vp)[1]
            return best_pool, (not inp[best_any])

        # every user's own ternary answers to the prefix, and the prefix-folded latent
        outs = np.stack([np.asarray(branch_of_per_user(E, rows, [q] * len(rows)), np.int64)
                         for q in prefix], 1)
        z = genv_latent(E, prefix, rows).numpy()
        lab_u = np.full(len(rows), -1, np.int64)
        src_u = np.zeros(len(rows), np.int64)                # 0 = marginal state, 1 = answer path
        sid_u = np.zeros(len(rows), np.int64)
        marg = [s for s in d["states"] if s["path"] == "*"]
        assert len(marg) == 1
        lm, lm_cat = _label(marg[0])
        n_cat_argmax += int(lm_cat); n_states_used += 1
        lab_u[:] = pool_index[cands[lm]]
        sid_u[:] = next_sid
        state_meta.append({"state_id": next_sid, "prefix_index": pi, "path": "*",
                           "label_pool_idx": int(pool_index[cands[lm]]),
                           "n_users_banked": int(marg[0]["n_users"])})
        next_sid += 1
        for s in d["states"]:
            if s["path"] == "*":
                continue
            li, lcat = _label(s)
            n_cat_argmax += int(lcat); n_states_used += 1
            ix = np.asarray([rowpos[int(r)] for r in s["rows"]], np.int64)
            # a user's OWN answer path is the more specific measurement; it wins over the marginal
            lab_u[ix] = pool_index[cands[li]]
            src_u[ix] = 1
            sid_u[ix] = next_sid
            state_meta.append({"state_id": next_sid, "prefix_index": pi, "path": s["path"],
                               "label_pool_idx": int(pool_index[cands[li]]),
                               "n_users_banked": int(s["n_users"])})
            next_sid += 1
            # the state's own answers must agree with the path key it was banked under
            key = "".join(str(int(x)) for x in outs[ix[0]])
            assert key == s["path"], (f"prefix {pi}: banked path {s['path']!r} does not reproduce "
                                      f"from branch_of_per_user ({key!r}) -- harness bug")
        hq = np.zeros((len(rows), MAX_K), np.int64)
        ho = np.zeros((len(rows), MAX_K), np.int64)
        for j, q in enumerate(prefix):
            hq[:, j] = pool_index[q]
            ho[:, j] = outs[:, j]
        Zs.append(z.astype(np.float32)); HQ.append(hq.astype(np.int16))
        HO.append(ho.astype(np.int8)); KK.append(np.full(len(rows), k, np.int8))
        LAB.append(lab_u.astype(np.int16)); PFX.append(np.full(len(rows), pi, np.int16))
        SRC.append(src_u.astype(np.int8)); SID.append(sid_u.astype(np.int32))
        prefix_meta.append({"prefix_index": pi, "k": k,
                            "prefix": [[int(a), int(b)] for a, b in prefix],
                            "n_states": len(d["states"]), "n_paths_kept": int(d["n_paths_kept"]),
                            "marginal_label_pool_idx": int(pool_index[cands[lm]]),
                            "n_users_on_a_kept_path": int(src_u.sum())})
        log(f"[prefix {pi:02d}] k={k} {len(d['states'])} states "
            f"({int(src_u.sum()):,} of 10,000 users on a measured answer path, the rest carry the "
            f"prefix's marginal label)")

    # ---- the other off-manifold states: full-pool below-floor + static-order --------------------
    for r in extra:
        rws = r["rows"] if r["rows"] is not None else [int(x) for x in rows]
        n = len(rws)
        prefix = r["prefix"]
        k = len(prefix)
        assert 0 < k <= MAX_K, f"{r['sid']}: prefix length {k}"
        if r["label_q"] is not None:
            lq = r["label_q"]
        else:
            inp = {q: g for q, g in r["gains"].items() if q in pool_index}
            assert inp, f"{r['sid']}: no in-pool candidate measured"
            lq = max(inp.items(), key=lambda kv: kv[1])[0]
        assert lq in pool_index, f"{r['sid']}: label {lq} is not in the 1,000-pool"
        assert lq not in set(prefix), f"{r['sid']}: the label is already in the prefix"
        outs = np.stack([np.asarray(branch_of_per_user(E, rws, [q] * n), np.int64)
                         for q in prefix], 1)
        z = genv_latent(E, prefix, rws).numpy()
        hq = np.zeros((n, MAX_K), np.int64); ho = np.zeros((n, MAX_K), np.int64)
        for j, q in enumerate(prefix):
            hq[:, j] = pool_index[q]; ho[:, j] = outs[:, j]
        Zs.append(z.astype(np.float32)); HQ.append(hq.astype(np.int16))
        HO.append(ho.astype(np.int8)); KK.append(np.full(n, k, np.int8))
        LAB.append(np.full(n, pool_index[lq], np.int16))
        PFX.append(np.full(n, -1 if r["kind"] == "static_order" else -2, np.int16))
        SRC.append(np.full(n, 2 if r["kind"] == "static_order" else 3, np.int8))
        SID.append(np.full(n, next_sid, np.int32))
        state_meta.append({"state_id": next_sid, "kind": r["kind"], "sid": r["sid"],
                           "prefix_len": k, "n_users": n,
                           "label_pool_idx": int(pool_index[lq]), "teacher": r["teacher"]})
        next_sid += 1; n_states_used += 1
    log(f"[extra] {len(extra)} extra off-manifold states folded")

    Z = np.concatenate(Zs); HQ = np.concatenate(HQ); HO = np.concatenate(HO)
    KK = np.concatenate(KK); LAB = np.concatenate(LAB).astype(np.int64)
    PFX = np.concatenate(PFX); SRC = np.concatenate(SRC); SID = np.concatenate(SID)
    np.savez(SEEDED_NPZ, z=Z, hist_q=HQ, hist_out=HO, k=KK, label=LAB.astype(np.int16),
             prefix=PFX, src=SRC, state_id=SID, rows=np.asarray(rows, np.int64))

    # ---- ★ THE COMPOSITION, PRINTED BEFORE ANY TIME IS SPENT -------------------------------------
    qs, mass = np.unique(LAB, return_counts=True)
    ch = np.asarray([pool[int(q)][0] for q in qs])
    tree_lab = np.unique(np.asarray(Dva["label_idx"], np.int64))
    comp = {
        "n_states": int(n_states_used), "n_prefixes": len(files),
        "n_examples": int(len(LAB)),
        "n_examples_from_a_measured_answer_path": int((SRC == 1).sum()),
        "n_examples_from_a_prefix_marginal_state": int((SRC == 0).sum()),
        "n_examples_from_a_static_order_state": int((SRC == 2).sum()),
        "n_examples_from_a_full_pool_below_floor_state": int((SRC == 3).sum()),
        "n_states_by_kind": {"random_prefix_marginal": 19,
                             "random_prefix_answer_path": int(next_sid - len(extra) - 19),
                             "static_order": sum(1 for r in extra if r["kind"] == "static_order"),
                             "deep_full_pool": sum(1 for r in extra if r["kind"] == "deep_full_pool")},
        "n_distinct_label_questions": int(len(qs)),
        "label_items": int((ch == 0).sum()), "label_concepts": int((ch == 1).sum()),
        "label_concept_share_example_weighted":
            float(np.mean([pool[int(q)][0] == 1 for q in LAB])),
        "label_mass_quantiles": [int(x) for x in np.percentile(mass, [0, 25, 50, 75, 100])],
        "n_label_questions_the_tree_also_uses": int(len(set(qs.tolist()) & set(tree_lab.tolist()))),
        "n_label_questions_the_tree_never_uses": int(len(set(qs.tolist()) - set(tree_lab.tolist()))),
        "prefix_length_histogram": {str(int(v)): int(c) for v, c in
                                    zip(*np.unique(KK, return_counts=True))},
        "n_states_whose_argmax_over_all_50_was_catalogue_wide": int(n_cat_argmax),
        "teacher": ("measured-NDCG argmax over the 20 IN-POOL candidates of the 50 that sweep "
                    "sampled (20 pool + 30 catalogue). NOT the argmax of the 1,000-pool."),
        "top_label_questions": [{"pool_idx": int(q), "channel": "item" if pool[int(q)][0] == 0
                                 else "concept", "id": int(pool[int(q)][1]), "n_examples": int(m)}
                                for q, m in sorted(zip(qs.tolist(), mass.tolist()),
                                                   key=lambda r: -r[1])[:15]]}
    log("=" * 100)
    log(f"[composition] {comp['n_states']} seeded states over {comp['n_prefixes']} prefixes -> "
        f"{comp['n_examples']:,} (user, state) examples "
        f"({comp['n_examples_from_a_measured_answer_path']:,} on a measured answer path, "
        f"{comp['n_examples_from_a_prefix_marginal_state']:,} on a prefix marginal)")
    log(f"[composition] {comp['n_distinct_label_questions']} distinct label questions "
        f"({comp['label_items']} items / {comp['label_concepts']} concepts); "
        f"{comp['n_label_questions_the_tree_never_uses']} of them are questions the TREE NEVER ASKS")
    log(f"[composition] label mass quantiles {comp['label_mass_quantiles']}; "
        f"concept share (example-weighted) {comp['label_concept_share_example_weighted']:.4f} "
        f"vs the tree's 0.5457")
    log(f"[composition] prefix lengths {comp['prefix_length_histogram']}")
    log(f"[composition] the argmax over all 50 candidates was a CATALOGUE question at "
        f"{comp['n_states_whose_argmax_over_all_50_was_catalogue_wide']} of {comp['n_states']} "
        f"states; those states use their in-pool argmax instead")
    for r in comp["top_label_questions"][:8]:
        log(f"    {r['channel']:>7} {r['id']:>5} pool_idx {r['pool_idx']:>4} "
            f"{r['n_examples']:>7,} examples")
    log("=" * 100)

    out = {"analysis": "policy_p2_seeded_states", "design_sheet":
           "docs/design/PAPER_B_POLICY_DESIGN.md I.2", "citable": True, "test_touched": False,
           "split": "validation", "composition": comp, "prefixes": prefix_meta,
           "states": state_meta,
           "npz": os.path.basename(SEEDED_NPZ), "canonical_snap": f0,
           "minutes": round((time.time() - t00) / 60, 2)}
    with open(SEEDED_JSON + ".tmp", "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    os.replace(SEEDED_JSON + ".tmp", SEEDED_JSON)
    log(f"[build] wrote {SEEDED_JSON} and {SEEDED_NPZ}")
    return out


# ============================================================================ the fine-tune
def seeded_batch(S, phi_t, idx, P, form):
    """One seeded batch in encoding `form` ('asked' or 'volunteered')."""
    B = len(idx)
    z = torch.from_numpy(np.ascontiguousarray(S["z"][idx]).astype(np.float32))
    T = TOTAL_L - 1
    hq = np.zeros((B, T), np.int64); ho = np.zeros((B, T), np.int64)
    hm = np.zeros((B, T), np.float32)
    asked = np.zeros((B, P), bool)
    k = np.asarray(S["k"][idx], np.int64)
    if form == "asked":
        sq = np.asarray(S["hist_q"][idx], np.int64); so = np.asarray(S["hist_out"][idx], np.int64)
        for j in range(min(MAX_K, T)):
            m = k > j
            if not m.any():
                continue
            hq[m, j] = sq[m, j]; ho[m, j] = so[m, j]; hm[m, j] = 1.0
        asked[np.repeat(np.arange(B), T)[hm.ravel() > 0], hq.ravel()[hm.ravel() > 0]] = True
        pos = np.stack([k, TOTAL_L - k], 1).astype(np.float32)
    else:
        # the encoding the evaluation presents: the belief has moved, nothing was asked
        pos = np.stack([np.zeros(B), np.full(B, TOTAL_L)], 1).astype(np.float32)
    y = np.asarray(S["label"][idx], np.int64)
    asked[np.arange(B), y] = False
    return (z, torch.from_numpy(phi_t[hq]), torch.from_numpy(ho), torch.from_numpy(hm),
            torch.from_numpy(pos), torch.from_numpy(asked), torch.from_numpy(y))


def finetune(args):
    from strategy_channel_suite import (build_greedy_env, contract_selftest, genv_eval, genv_mean)
    from ndcg_tree import build_pool, leak_assert, branch_of_selftest
    torch.manual_seed(SEED); np.random.seed(SEED)
    rng = np.random.default_rng(SEED)
    t00 = time.time()
    os.makedirs(P2_DIR, exist_ok=True)

    zf = np.load(SEEDED_NPZ)
    S = {k: zf[k] for k in zf.files}
    meta = json.load(open(SEEDED_JSON))
    Dtr = load_traj(train=True); Dva = load_traj(train=False)
    pool = Dtr["pool"]; P = len(pool)

    E = build_greedy_env(args.snapshot, "validation", args.concept_ckpt, f"p2/{args.arm}")
    contract_selftest(E, E["rows"]); leak_assert(E, E["rows"], f"p2/{args.arm}")
    rows = E["rows"]
    f0, t0_ = genv_mean(E, [], rows)
    assert round(f0, 8) == VAL_M0_FULL, f"CANONICAL SNAP FAILED: {f0!r}"
    log(f"[gate] canonical m=0 snap full={f0!r} -- exact")
    pool_e, _i, _c = build_pool(E, 500, 500)
    assert pool_e == pool, "the env's pool is not the trajectory bank's pool"
    assert np.array_equal(np.asarray(S["rows"], np.int64), np.asarray(rows, np.int64)), \
        "the seeded states were built on different rows"
    branch_of_selftest(E, rows[:2000], [pool[0], pool[500], (1, 114)])
    static_seq = [(int(a), int(b)) for a, b in json.load(open(STATIC_JSON))["sequence"]]
    sf, stl = genv_eval(E, static_seq, rows)
    assert abs(float(np.nanmean(sf)) - STATIC_VAL_FULL) < 1e-12, "BANKED-HEADLINE ASSERT FAILED"
    log(f"[gate] static comparator reproduces EXACTLY: {float(np.nanmean(sf))!r}")

    phi_t, norm = build_phi(E, pool)
    phi_g = torch.from_numpy(phi_t)
    blob = torch.load(args.init_ckpt, map_location="cpu")
    assert [(int(a), int(b)) for a, b in blob["pool"]] == pool
    assert np.array_equal(blob["norm_mu"], norm[0]) and np.array_equal(blob["norm_sd"], norm[1])
    pol = Policy(E["ctx"].d, phi_t.shape[1], P, arch=blob["arch"], state=blob["state"],
                 use_pos=blob["use_pos"])
    pol.load_state_dict(blob["model"])
    log(f"[init] fine-tuning from {os.path.basename(args.init_ckpt)} epoch {blob['epoch']} "
        f"val_full {blob['val_full']:.6f} -- the E1 anchor")

    lab_seed = np.asarray(S["label"], np.int64).copy()
    if args.arm == "shuffle":
        # HARD RULE 5: a state gets ANOTHER STATE's measured label. The label marginal over states
        # survives; the state->label association is destroyed. The replay is untouched.
        sid = np.asarray(S["state_id"], np.int64)
        uk = np.unique(sid)
        lab_of = {int(s_): int(lab_seed[np.flatnonzero(sid == s_)[0]]) for s_ in uk}
        for s_ in uk:                       # a state's label is one value; assert it before permuting
            assert len(np.unique(lab_seed[sid == s_])) == 1, "a seeded state carries two labels"
        perm = uk[rng.permutation(len(uk))]
        rep = {int(a): int(b) for a, b in zip(uk, perm)}
        lab_seed = np.asarray([lab_of[rep[int(s_)]] for s_ in sid], np.int64)
        log(f"[arm] SHUFFLE CONTROL: {len(uk)} seeded state-groups permuted; "
            f"{float((lab_seed != np.asarray(S['label'], np.int64)).mean()) * 100:.1f}% of seeded "
            f"examples now carry another state's label")
    S["label"] = lab_seed

    opt = torch.optim.Adam(pol.parameters(), lr=args.lr)
    lossf = nn.CrossEntropyLoss()
    n_seed_b = int(round(args.batch * SEEDED_FRAC))
    n_rep_b = args.batch - n_seed_b
    n_ex = len(Dtr["uid"]) * TOTAL_L
    log(f"[mixture] batch {args.batch} = {n_rep_b} replay (E1 tree labels) + {n_seed_b} seeded "
        f"({SEEDED_FRAC:.4f} declared constant); seeded examples drawn half 'asked', half "
        f"'volunteered'")

    stem = f"p2_{args.arm}"
    ckpt = os.path.join(P2_DIR, stem + ".pt")
    hist = []
    passed = None
    for ep in range(1, args.epochs + 1):
        pol.train()
        te = time.time()
        tot = seen = hit = 0.0
        for si in range(args.steps):
            e = rng.integers(0, n_ex, size=n_rep_b)
            u = (e // TOTAL_L).astype(np.int64); t = (e % TOTAL_L).astype(np.int64)
            z, hp, ho, hm, pos, am, yy = make_batch(Dtr, phi_t, u, t, P)
            si_idx = np.sort(rng.integers(0, len(S["label"]), size=n_seed_b))
            half = n_seed_b // 2
            parts = [seeded_batch(S, phi_t, si_idx[:half], P, "asked"),
                     seeded_batch(S, phi_t, si_idx[half:], P, "volunteered")]
            z = torch.cat([z] + [p[0] for p in parts])
            hp = torch.cat([hp] + [p[1] for p in parts])
            ho = torch.cat([ho] + [p[2] for p in parts])
            hm = torch.cat([hm] + [p[3] for p in parts])
            pos = torch.cat([pos] + [p[4] for p in parts])
            am = torch.cat([am] + [p[5] for p in parts])
            yy = torch.cat([yy] + [p[6] for p in parts])
            s = pol(z, hp, ho, hm, pos, phi_g, am)
            loss = lossf(s, yy)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * len(yy); seen += len(yy)
            hit += int((s.argmax(1) == yy).sum())
            if (si + 1) % 250 == 0:
                log(f"[ep{ep} {si + 1}/{args.steps}] loss {tot / seen:.4f} top1 {hit / seen:.4f} "
                    f"({(time.time() - te) / 60:.1f}m)")
        # --- what the epoch bought, on both halves --------------------------------------------
        ag = teacher_forced_agreement(pol, phi_t, Dva, P)
        ro = rollout(E, pol, phi_t, pool, rows)
        vf, vt = ro["curves"][TOTAL_L]
        sd = {}
        pol.eval()
        for form in ("asked", "volunteered"):
            h = 0
            with torch.no_grad():
                for st in range(0, len(S["label"]), 4096):
                    ix = np.arange(st, min(st + 4096, len(S["label"])))
                    b = seeded_batch(S, phi_t, ix, P, form)
                    ss = pol(b[0], b[1], b[2], b[3], b[4], phi_g, b[5])
                    h += int((ss.argmax(1) == b[6]).sum())
            sd[form] = h / len(S["label"])
        rec = {"epoch": ep, "train_loss": tot / seen, "train_top1": hit / seen,
               "val_top1_vs_tree_teacher_forced": ag["top1"],
               "val_full_unseeded": float(np.nanmean(vf)), "val_tail_unseeded": float(np.nanmean(vt)),
               "seeded_top1_asked": sd["asked"], "seeded_top1_volunteered": sd["volunteered"],
               "spanning_gate_pass": bool(float(np.nanmean(vf)) >= E1_VAL_FULL),
               "n_distinct_interviews": len({tuple(r) for r in ro["asked"]}),
               "minutes": round((time.time() - te) / 60, 2)}
        hist.append(rec)
        log(f"[ep{ep}] loss {rec['train_loss']:.4f} | VAL unseeded full {rec['val_full_unseeded']:.5f} "
            f"(E1 {E1_VAL_FULL:.5f}, gate {'PASS' if rec['spanning_gate_pass'] else 'FAIL'}) | "
            f"seeded top1 asked {sd['asked']:.4f} volunteered {sd['volunteered']:.4f} | "
            f"tree agreement {ag['top1']:.4f} | {rec['n_distinct_interviews']} interviews | "
            f"{rec['minutes']:.1f}m")
        # ALWAYS bank the last epoch under its own name. This does NOT change the declared selection
        # rule -- `p2_<arm>.pt` is still written only when the spanning gate is cleared. It exists so
        # that an arm the gate REJECTS can still be evaluated as a declared diagnostic rather than
        # disappearing: "trained on seeded states, deviations still cost" is the pre-registered
        # falsifier and it must remain measurable whether or not the arm is admissible.
        torch.save({"model": pol.state_dict(), "arch": blob["arch"], "state": blob["state"],
                    "use_pos": blob["use_pos"], "d_phi": int(phi_t.shape[1]), "n_pool": P,
                    "epoch": ep, "val_full": rec["val_full_unseeded"],
                    "val_tail": rec["val_tail_unseeded"], "norm_mu": norm[0], "norm_sd": norm[1],
                    "arm": f"p2_{args.arm}_lastepoch",
                    "pool": [[int(a), int(b)] for a, b in pool], "seed": SEED},
                   os.path.join(P2_DIR, stem + "_last.pt"))
        if rec["spanning_gate_pass"]:
            # DECLARED SELECTION RULE: the LAST epoch that clears the spanning gate.
            torch.save({"model": pol.state_dict(), "arch": blob["arch"], "state": blob["state"],
                        "use_pos": blob["use_pos"], "d_phi": int(phi_t.shape[1]), "n_pool": P,
                        "epoch": ep, "val_full": rec["val_full_unseeded"],
                        "val_tail": rec["val_tail_unseeded"], "norm_mu": norm[0],
                        "norm_sd": norm[1], "arm": f"p2_{args.arm}",
                        "pool": [[int(a), int(b)] for a, b in pool], "seed": SEED}, ckpt)
            passed = ep
            log(f"[ckpt] epoch {ep} clears the spanning gate -> {os.path.basename(ckpt)}")
        out = {"analysis": "policy_p2_finetune", "design_sheet":
               "docs/design/PAPER_B_POLICY_DESIGN.md I.2", "citable": True, "test_touched": False,
               "arm": args.arm, "lr": args.lr, "epochs": args.epochs, "steps_per_epoch": args.steps,
               "batch": args.batch, "seeded_frac": SEEDED_FRAC,
               "seeded_composition": meta["composition"], "history": hist,
               "selected_epoch": passed,
               "selection_rule": "the LAST epoch clearing the spanning gate (val_full_unseeded >= "
                                 f"{E1_VAL_FULL}), declared before the run",
               "init_from": os.path.basename(args.init_ckpt),
               "minutes": round((time.time() - t00) / 60, 2)}
        p = os.path.join(P2_DIR, stem + "_history.json")
        with open(p + ".tmp", "w") as fh:
            json.dump(out, fh, indent=1, default=float)
        os.replace(p + ".tmp", p)
    if passed is None:
        log("[REJECT] no epoch cleared the spanning gate -- this arm is REJECTED (design sheet I.2)")
    return out


# ============================================================================ the evaluation
def evaluate(args):
    from strategy_channel_suite import (build_greedy_env, contract_selftest, genv_eval, genv_mean,
                                        genv_eval_multi)
    from ndcg_tree import build_pool, leak_assert, branch_of_per_user, branch_of_selftest
    t00 = time.time()
    os.makedirs(P2_DIR, exist_ok=True)
    torch.manual_seed(SEED); np.random.seed(SEED)

    Dva = load_traj(train=False)
    pool = Dva["pool"]; P = len(pool)
    E = build_greedy_env(args.snapshot, "validation", args.concept_ckpt, "p2/eval")
    contract_selftest(E, E["rows"]); leak_assert(E, E["rows"], "p2/eval")
    rows = E["rows"]
    f0, t0_ = genv_mean(E, [], rows)
    assert round(f0, 8) == VAL_M0_FULL, f"CANONICAL SNAP FAILED: {f0!r}"
    log(f"[gate] canonical m=0 snap full={f0!r} -- exact")
    pool_e, _i, _c = build_pool(E, 500, 500)
    assert pool_e == pool, "the env's pool is not the trajectory bank's pool"
    branch_of_selftest(E, rows[:2000], [pool[0], pool[500], (1, 114)])
    static_seq = [(int(a), int(b)) for a, b in json.load(open(STATIC_JSON))["sequence"]]
    sf, stl = genv_eval(E, static_seq, rows)
    assert abs(float(np.nanmean(sf)) - STATIC_VAL_FULL) < 1e-12, "BANKED-HEADLINE ASSERT FAILED"
    z1 = np.load(E1_PER_USER)
    e1_full = np.asarray(z1["policy_full"], np.float64)
    e1_asked = np.asarray(z1["asked"], np.int64)
    assert np.array_equal(e1_asked, np.asarray(Dva["label_idx"], np.int64)), \
        "E1's banked rollout is not the tree's realised interview"
    log(f"[gate] E1 == the tree, banked mean full {float(e1_full.mean()):.9f}")

    phi_t, norm = build_phi(E, pool)
    tree_asked = np.asarray(Dva["label_idx"], np.int64)
    tree_outs = np.asarray(Dva["outcome"], np.int64)
    static_idx = np.asarray([pool.index(q) for q in static_seq], np.int64)
    static_asked = np.tile(static_idx[None, :], (len(rows), 1))

    pols, unseeded = {}, {}
    for spec in args.ckpts.split(","):
        name, path = spec.split("=", 1)
        b = torch.load(path, map_location="cpu")
        assert [(int(a), int(bb)) for a, bb in b["pool"]] == pool
        assert np.array_equal(b["norm_mu"], norm[0]) and np.array_equal(b["norm_sd"], norm[1])
        pl = Policy(E["ctx"].d, phi_t.shape[1], P, arch=b["arch"], state=b["state"],
                    use_pos=b["use_pos"])
        pl.load_state_dict(b["model"]); pl.eval()
        pols[name] = pl
        ro = rollout(E, pl, phi_t, pool, rows)
        uf, ut = ro["curves"][TOTAL_L]
        unseeded[name] = {"asked": ro["asked"], "outs": ro["outs"], "full": uf, "tail": ut,
                          "val_full": float(np.nanmean(uf)), "val_tail": float(np.nanmean(ut)),
                          "epoch": int(b["epoch"]),
                          "spanning_gate_pass": bool(float(np.nanmean(uf)) >= E1_VAL_FULL),
                          "deviates_from_tree_frac": float((ro["asked"] != tree_asked).any(1).mean()),
                          "n_distinct_interviews": len({tuple(r) for r in ro["asked"]})}
        log(f"[{name}] UNSEEDED val full {unseeded[name]['val_full']:.5f} vs E1 {E1_VAL_FULL:.5f} "
            f"-- spanning gate {'PASS' if unseeded[name]['spanning_gate_pass'] else 'FAIL'}; "
            f"deviates from the tree on {unseeded[name]['deviates_from_tree_frac'] * 100:.2f}% of "
            f"users unseeded; {unseeded[name]['n_distinct_interviews']} distinct interviews")

    results = {}
    for k in [int(x) for x in args.ks.split(",")]:
        log("=" * 100)
        log(f"=== k = {k} volunteered items, FRESH draw seed {args.draw_seed} "
            f"(R/R2 used {SEED}; the two draws are disjoint by construction)")
        vol, draw_info = PR.draw_volunteered(E, rows, k, seed=args.draw_seed)
        draw_info["seed"] = args.draw_seed
        draw_info["disjoint_from"] = f"R/R2's draw at seed {SEED}"
        cells = {}

        def _fixed(name, asked_arr):
            iseq, cseq, redun = PR.build_seqs(asked_arr, pool, vol)
            f, t = genv_eval_multi(E, rows, iseq, cseq)
            outs = np.stack([np.asarray(branch_of_per_user(
                E, rows, [pool[int(j)] for j in asked_arr[:, tt]]), np.int64)
                for tt in range(asked_arr.shape[1])], 1)
            cells[name] = {"full": f, "tail": t, "asked": asked_arr, "outs": outs, "redun": redun}
            log(f"  [{name}] full {float(np.nanmean(f)):.5f} tail {float(np.nanmean(t)):.5f}")

        empty = [[] for _ in rows]
        fn, tn = genv_eval_multi(E, rows, vol, empty)
        cells["none"] = {"full": fn, "tail": tn, "asked": np.full((len(rows), 0), -1, np.int64),
                         "outs": np.full((len(rows), 0), -1, np.int64),
                         "redun": np.zeros(len(rows), np.int64)}
        log(f"  [none] full {float(np.nanmean(fn)):.5f}")
        _fixed("static", static_asked)
        _fixed("tree", tree_asked)
        for name in pols:
            _fixed(name + "_frozen", unseeded[name]["asked"])
            t0 = time.time()
            ro = PR.rollout_seeded(E, pols[name], phi_t, pool, rows, vol)
            f, t = ro["curves"][TOTAL_L]
            _i, _c, redun = PR.build_seqs(ro["asked"], pool, vol)
            cells[name] = {"full": f, "tail": t, "asked": ro["asked"],
                           "outs": ro["outs"].astype(np.int64), "redun": redun}
            log(f"  [{name}] full {float(np.nanmean(f)):.5f} tail {float(np.nanmean(t)):.5f} "
                f"({(time.time() - t0) / 60:.1f}m)")

        gf = {a: np.asarray(cells[a]["full"], np.float64) for a in cells}
        gt = {a: np.asarray(cells[a]["tail"], np.float64) for a in cells}
        red = {}
        for a in cells:
            ak = cells[a]["asked"]
            if ak.shape[1] == 0:
                continue
            niq = np.asarray([sum(1 for j in r if pool[int(j)][0] == 0) for r in ak], np.float64)
            rr = np.asarray(cells[a]["redun"], np.float64)
            red[a] = {"mean_item_questions_of_16": float(niq.mean()),
                      "mean_reasks_per_user": float(rr.mean()),
                      "reasks_per_item_question": float(rr.mean() / niq.mean()),
                      "vs_tree_paired": None}
        for a in red:
            if a == "tree":
                continue
            d = (np.asarray(cells[a]["redun"], np.float64) -
                 np.asarray(cells["tree"]["redun"], np.float64))
            red[a]["vs_tree_paired"] = paired_bootstrap(d)
        comp, adapt = {}, {}
        for name in pols:
            comp[f"{name}_minus_its_own_frozen_order"] = {
                "full": paired_bootstrap(gf[name] - gf[name + "_frozen"]),
                "tail": paired_bootstrap(gt[name] - gt[name + "_frozen"]),
                "why": "★ THE DEVIATION-COST FLIP. Same user, same fold, same volunteered evidence; "
                       "ONLY the questions differ. R2 measured -0.01246/-0.01234/-0.00860.",
                "R2_belief_only_reference": {"2": -0.01246, "4": -0.01234, "8": -0.00860}[str(k)]
                if str(k) in ("2", "4", "8") else None}
            comp[f"{name}_minus_tree"] = {"full": paired_bootstrap(gf[name] - gf["tree"]),
                                          "tail": paired_bootstrap(gt[name] - gt["tree"])}
            comp[f"{name}_minus_static"] = {"full": paired_bootstrap(gf[name] - gf["static"]),
                                            "tail": paired_bootstrap(gt[name] - gt["static"])}
            comp[f"{name}_minus_none"] = {"full": paired_bootstrap(gf[name] - gf["none"]),
                                          "tail": paired_bootstrap(gt[name] - gt["none"])}
            d = cells[name]["asked"] != unseeded[name]["asked"]
            dev = d.any(1)
            adapt[name] = {
                "frac_never_deviates_from_its_own_unseeded_rollout": float((~dev).mean()),
                "mean_positions_differing": float(d.sum(1).mean()),
                "n_deviators": int(dev.sum()),
                "frac_users_differing_at_each_position": [float(d[:, tt].mean())
                                                          for tt in range(TOTAL_L)],
                "among_deviators_vs_own_frozen_order":
                    paired_bootstrap((gf[name] - gf[name + "_frozen"])[dev]) if dev.any() else None,
                "n_distinct_interviews": len({tuple(r) for r in cells[name]["asked"]}),
                "pep_divergence": pep_divergence_from_arrays(cells[name]["asked"],
                                                            cells[name]["outs"]),
                "concept_share": channel_mix(cells[name]["asked"], pool),
                "answer_rate": float((cells[name]["outs"] != PN.OUTCOME["noanswer"]).mean())}
        comp["tree_minus_static"] = {"full": paired_bootstrap(gf["tree"] - gf["static"]),
                                     "tail": paired_bootstrap(gt["tree"] - gt["static"])}
        comp["tree_minus_none"] = {"full": paired_bootstrap(gf["tree"] - gf["none"]),
                                   "tail": paired_bootstrap(gt["tree"] - gt["none"])}
        results[str(k)] = {"draw": draw_info,
                           "scores": {a: {"full": float(np.nanmean(gf[a])),
                                          "tail": float(np.nanmean(gt[a]))} for a in cells},
                           "comparisons": comp, "adaptation": adapt,
                           "redundancy_per_item_question": red}
        np.savez_compressed(os.path.join(P2_DIR, f"p2_eval_k{k}_per_user.npz"),
                            **{f"{a}_full": cells[a]["full"] for a in cells},
                            **{f"{a}_tail": cells[a]["tail"] for a in cells},
                            **{f"{a}_asked": cells[a]["asked"] for a in cells},
                            **{f"{a}_redun": cells[a]["redun"] for a in cells},
                            rows=np.asarray(rows, np.int64))
        out = {"analysis": "policy_p2_eval", "design_sheet":
               "docs/design/PAPER_B_POLICY_DESIGN.md I.2", "citable": True, "test_touched": False,
               "split": "validation", "n_users": len(rows), "draw_seed": args.draw_seed,
               "teacher_honesty": ("the seeded-state labels this policy was fine-tuned on are the "
                                   "measured-NDCG argmax over the 20 IN-POOL candidates of the 50 "
                                   "that sweep sampled, NOT over the 1,000-pool"),
               "unseeded": {n: {kk: vv for kk, vv in unseeded[n].items()
                                if kk not in ("asked", "outs", "full", "tail")} for n in unseeded},
               "E1_reference": {"val_full": E1_VAL_FULL, "static_full": float(np.nanmean(sf))},
               "by_k": results, "minutes": round((time.time() - t00) / 60, 2)}
        p = os.path.join(P2_DIR, "p2_eval.json")
        with open(p + ".tmp", "w") as fh:
            json.dump(out, fh, indent=1, default=float)
        os.replace(p + ".tmp", p)
        for name in pols:
            c = comp[f"{name}_minus_its_own_frozen_order"]["full"]
            log(f"[k={k}] ★ {name} - its own frozen order: {c['mean']:+.5f} "
                f"[{c['lo']:+.5f}, {c['hi']:+.5f}] (R2 belief-only: "
                f"{comp[f'{name}_minus_its_own_frozen_order']['R2_belief_only_reference']}); "
                f"adapts on {(1 - adapt[name]['frac_never_deviates_from_its_own_unseeded_rollout']) * 100:.2f}% "
                f"of users, {adapt[name]['mean_positions_differing']:.2f} positions")
    log(f"[done] {round((time.time() - t00) / 60, 2)} minutes")
    return out


def main():
    ap = argparse.ArgumentParser(description="P2 (Part I.2): the evidence-seeded fine-tune")
    ap.add_argument("--mode", required=True, choices=["build", "train", "eval"])
    ap.add_argument("--arm", default="main", choices=["main", "shuffle"])
    ap.add_argument("--init_ckpt", default=os.path.join(OUTDIR, "policy_main_equivariant_both.pt"))
    ap.add_argument("--ckpts", default="", help="name=path,... for --mode eval")
    ap.add_argument("--ks", default="2,4,8")
    ap.add_argument("--draw_seed", type=int, default=DRAW_SEED)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--steps", type=int, default=1000, help="optimiser steps per epoch")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-4, help="the phase-3 fine-tune rate")
    ap.add_argument("--snapshot", default=None)
    ap.add_argument("--concept_ckpt", default=None)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--full_threads", action="store_true")
    ap.add_argument("--i_have_author_goahead", action="store_true")
    args = ap.parse_args()
    assert args.i_have_author_goahead, "REFUSING TO RUN without --i_have_author_goahead"
    from strategy_channel_suite import CERT_TOWER, CERT_CONC
    args.snapshot = args.snapshot or CERT_TOWER
    args.concept_ckpt = args.concept_ckpt or CERT_CONC
    log("#" * 110)
    log("# P2 -- GENERALISATION ACROSS STATES (design sheet Part I.2). VALIDATION ONLY.")
    log("# The 10,000 canonical TEST users are NOT built, NOT loaded, NOT touched.")
    log("#" * 110)
    return {"build": build_seeded, "train": finetune, "eval": evaluate}[args.mode](args)


if __name__ == "__main__":
    main()
