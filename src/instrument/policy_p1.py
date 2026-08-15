r"""P1 -- GENERALISATION ACROSS QUESTIONS: does the policy emit a question it was never taught?

Design sheet: `docs/design/PAPER_B_POLICY_DESIGN.md` **PART I.1 is binding**. Committed before any
run (HARD RULE 10). Validation only for every measurement; the 10,000 canonical TEST users are never
built, never loaded, never named -- there is no `--split` in this file.

WHAT IS BEING ASKED
-------------------
The tree's labels use 172 distinct questions. Hold out ~20% of them BY IDENTITY, retrain E1 from
scratch on the remaining labels, and leave the held-out questions IN the candidate table so they stay
*emittable* while never being *supervised*. Then, at exactly the (user, position) states whose true
label was a held-out question, ask whether the retrained policy emits it anyway.

THE TWO MODES
-------------
  --mode split   computes the held-out set and the k=5 neighbour lists and writes them to disk.
                 ★ THIS RUNS AND IS COMMITTED BEFORE ANY TRAINING (design sheet I.1 measurement 2:
                 "neighbour list computed and committed BEFORE training"). It prints the composition
                 of the held-out set -- channel split, label-mass histogram, and how many
                 (user, position) examples each held-out question carries -- before any time is spent.
  --mode audit   loads trained checkpoints and measures emission at the held-out states.

THE STRATIFICATION, DECLARED (HARD RULE 1)
------------------------------------------
Channel (item / concept) x label-mass QUARTILE, where label mass = the number of TRAIN (user,
position) examples carrying that question as its label. Stratifying on mass is not cosmetic: the mass
distribution is 2,948 ... 140,768 (the root concept alone carries every user at position 0), so an
unstratified 20% draw would land almost entirely on rare labels and the experiment would measure
nothing. 20% is held out inside each of the 8 strata, seed 98765.

THE DUMB COMPARATOR, DECLARED
-----------------------------
A popularity-ranked emitter. "Popularity" is put on ONE scale across the two channels as *the number
of train users who can answer this question*: `ctx.cnt[i]` for an item (train rating count), and
`answer_rate[c] * 140,768` for a concept. If that fixed ranking finds the held-out question as often
as the trained policy does, similarity added nothing and the claim fails.
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
import torch

import policy_net as PN
from policy_net import (OUTDIR, SEED, TOTAL_L, STATIC_JSON, STATIC_VAL_FULL, VAL_M0_FULL,
                        Policy, build_phi, paired_bootstrap, load_traj, make_batch, log)

P1_DIR = os.path.join(OUTDIR, "p1_heldout")
SPLIT_JSON = os.path.join(P1_DIR, "p1_split.json")
K_NN = 5
HELDOUT_FRAC = 0.20
N_MASS_STRATA = 4


# ============================================================================ the split
def build_split(args):
    """Held-out question identities + the k=5 neighbour lists, written BEFORE any training."""
    os.makedirs(P1_DIR, exist_ok=True)
    from strategy_channel_suite import (build_greedy_env, contract_selftest, genv_eval, genv_mean,
                                        CERT_TOWER, CERT_CONC, CONCSTATS, N_TRAIN_USERS)
    from ndcg_tree import build_pool, leak_assert
    t00 = time.time()

    Dtr = load_traj(train=True)
    Dva = load_traj(train=False)
    pool = Dtr["pool"]
    lab_tr = np.asarray(Dtr["label_idx"], np.int64)
    lab_va = np.asarray(Dva["label_idx"], np.int64)
    qs, mass = np.unique(lab_tr, return_counts=True)
    assert set(qs.tolist()) == set(np.unique(lab_va).tolist()), \
        "the train and validation label question sets differ -- they were identical in E1"
    log(f"[labels] {len(qs)} distinct label questions behind {lab_tr.size:,} train examples")

    # ---- gates first, before anything is trusted -------------------------------------------------
    E = build_greedy_env(args.snapshot, "validation", args.concept_ckpt, "p1/split")
    contract_selftest(E, E["rows"])
    leak_assert(E, E["rows"], "p1/split")
    rows = E["rows"]
    assert len(rows) == 10000
    f0, t0_ = genv_mean(E, [], rows)
    assert round(f0, 8) == VAL_M0_FULL, f"CANONICAL SNAP FAILED: {f0!r} != {VAL_M0_FULL}"
    log(f"[gate] canonical m=0 snap full={f0!r} tail={t0_!r} -- exact")
    pool_e, _ip, _cp = build_pool(E, 500, 500)
    assert pool_e == pool, "the env's pool is not the trajectory bank's pool"
    static_seq = [(int(a), int(b)) for a, b in json.load(open(STATIC_JSON))["sequence"]]
    sf, _st = genv_eval(E, static_seq, rows)
    assert abs(float(np.nanmean(sf)) - STATIC_VAL_FULL) < 1e-12, "BANKED-HEADLINE ASSERT FAILED"
    log(f"[gate] greedy_static_policyB_combined reproduces EXACTLY: {float(np.nanmean(sf))!r}")

    phi_t, norm = build_phi(E, pool)
    log(f"[phi] {phi_t.shape[1]}-dim frozen descriptors for {len(pool)} candidates")

    # ---- the stratified draw ---------------------------------------------------------------------
    ch = np.asarray([pool[int(q)][0] for q in qs], np.int64)
    rng = np.random.default_rng(SEED)
    strata, heldout = {}, []
    for c in (0, 1):
        m = ch == c
        idx = np.flatnonzero(m)
        # mass quartiles WITHIN the channel, so the two channels are stratified on their own scales
        order = idx[np.argsort(mass[idx], kind="stable")]
        chunks = np.array_split(order, N_MASS_STRATA)
        for s, chunk in enumerate(chunks):
            n_take = int(round(len(chunk) * HELDOUT_FRAC))
            n_take = max(n_take, 1) if len(chunk) else 0
            take = rng.choice(len(chunk), size=n_take, replace=False) if n_take else []
            sel = [int(chunk[j]) for j in np.sort(np.asarray(take, np.int64))] if n_take else []
            strata[f"ch{c}_mass{s}"] = {
                "channel": "item" if c == 0 else "concept", "mass_stratum": s,
                "n_questions": len(chunk), "n_held_out": len(sel),
                "mass_range": [int(mass[chunk].min()), int(mass[chunk].max())] if len(chunk) else None,
                "held_out_pool_idx": [int(qs[j]) for j in sel]}
            heldout += [int(qs[j]) for j in sel]
    heldout = sorted(set(heldout))
    hset = set(heldout)
    log(f"[split] {len(heldout)} of {len(qs)} label questions held out "
        f"({len(heldout) / len(qs) * 100:.1f}%)")

    # ---- the neighbour lists, in the FROZEN descriptor space -------------------------------------
    # cosine similarity over the standardised descriptor. Computed for every held-out question
    # against every OTHER pool candidate (including other held-out ones -- the neighbour list is a
    # property of the descriptor space, not of the split).
    X = phi_t / (np.linalg.norm(phi_t, axis=1, keepdims=True) + 1e-12)
    S = X @ X.T
    np.fill_diagonal(S, -np.inf)
    nn = {}
    for q in heldout:
        o = np.argsort(-S[q])[:K_NN]
        nn[str(q)] = {"pool_idx": [int(j) for j in o],
                      "cos": [float(S[q, j]) for j in o],
                      "channel": ["item" if pool[int(j)][0] == 0 else "concept" for j in o],
                      "n_neighbours_in_label_set": int(sum(1 for j in o if int(j) in set(qs.tolist()))),
                      "n_neighbours_also_held_out": int(sum(1 for j in o if int(j) in hset))}
    same_ch = np.mean([1.0 if all(c == ("item" if pool[q][0] == 0 else "concept")
                                  for c in nn[str(q)]["channel"]) else 0.0 for q in heldout])
    log(f"[nn] k={K_NN} neighbour lists built; {same_ch * 100:.1f}% of held-out questions have an "
        f"all-same-channel neighbourhood")

    # ---- ★ THE COMPOSITION, PRINTED BEFORE ANY TIME IS SPENT -------------------------------------
    massmap = {int(q): int(m) for q, m in zip(qs, mass)}
    ho_mass = np.asarray([massmap[q] for q in heldout], np.int64)
    va_q, va_mass = np.unique(lab_va, return_counts=True)
    vamap = {int(q): int(m) for q, m in zip(va_q, va_mass)}
    ho_va = np.asarray([vamap.get(q, 0) for q in heldout], np.int64)
    n_ho_ex_tr = int(np.isin(lab_tr, heldout).sum())
    n_ho_ex_va = int(np.isin(lab_va, heldout).sum())
    per_pos_va = [int(np.isin(lab_va[:, t], heldout).sum()) for t in range(TOTAL_L)]
    comp = {
        "n_questions_total": int(len(qs)), "n_held_out": len(heldout),
        "held_out_items": int(sum(1 for q in heldout if pool[q][0] == 0)),
        "held_out_concepts": int(sum(1 for q in heldout if pool[q][0] == 1)),
        "kept_items": int((ch == 0).sum()) - int(sum(1 for q in heldout if pool[q][0] == 0)),
        "kept_concepts": int((ch == 1).sum()) - int(sum(1 for q in heldout if pool[q][0] == 1)),
        "train_examples_total": int(lab_tr.size),
        "train_examples_carrying_a_held_out_label": n_ho_ex_tr,
        "train_examples_dropped_frac": n_ho_ex_tr / lab_tr.size,
        "val_examples_total": int(lab_va.size),
        "val_examples_carrying_a_held_out_label": n_ho_ex_va,
        "val_held_out_examples_by_position": per_pos_va,
        "held_out_label_mass_train_quantiles": [int(x) for x in
                                                np.percentile(ho_mass, [0, 25, 50, 75, 100])],
        "kept_label_mass_train_quantiles": [int(x) for x in np.percentile(
            [massmap[int(q)] for q in qs if int(q) not in hset], [0, 25, 50, 75, 100])],
        "per_held_out_question": sorted(
            [{"pool_idx": int(q), "channel": "item" if pool[q][0] == 0 else "concept",
              "id": int(pool[q][1]), "train_label_mass": massmap[q],
              "val_label_mass": vamap.get(q, 0)} for q in heldout],
            key=lambda r: -r["train_label_mass"])}
    log("=" * 100)
    log(f"[composition] held out {comp['n_held_out']} questions "
        f"({comp['held_out_items']} items / {comp['held_out_concepts']} concepts); kept "
        f"{comp['kept_items']} items / {comp['kept_concepts']} concepts")
    log(f"[composition] train label mass of the held-out set, quantiles "
        f"{comp['held_out_label_mass_train_quantiles']} vs kept "
        f"{comp['kept_label_mass_train_quantiles']}")
    log(f"[composition] {n_ho_ex_tr:,} of {lab_tr.size:,} train examples "
        f"({n_ho_ex_tr / lab_tr.size * 100:.2f}%) carry a held-out label and are DROPPED from "
        f"training; {n_ho_ex_va:,} of {lab_va.size:,} validation examples are the states P1 measures "
        f"emission at")
    log(f"[composition] held-out validation states by position: {per_pos_va}")
    for r in comp["per_held_out_question"][:10]:
        log(f"    {r['channel']:>7} {r['id']:>5} pool_idx {r['pool_idx']:>4} "
            f"train mass {r['train_label_mass']:>7,} val mass {r['val_label_mass']:>6,}")
    log("=" * 100)

    # ---- popularity, the dumb comparator's ranking -----------------------------------------------
    zc = np.load(CONCSTATS)
    c_ar = (zc["n"] / N_TRAIN_USERS).astype(np.float64)
    cnt = np.asarray(E["ctx"].cnt, np.float64)
    popv = np.asarray([cnt[int(i)] if int(c) == 0 else c_ar[int(i)] * N_TRAIN_USERS
                       for c, i in pool], np.float64)

    out = {"analysis": "policy_p1_split", "design_sheet":
           "docs/design/PAPER_B_POLICY_DESIGN.md I.1", "citable": True, "test_touched": False,
           "split": "validation (measurement) / train (fit states)", "seed": SEED,
           "heldout_frac": HELDOUT_FRAC, "n_mass_strata": N_MASS_STRATA, "k_nn": K_NN,
           "pool": [[int(a), int(b)] for a, b in pool],
           "pool_contract": E["pool_contract"], "tower": E["tower"], "conc_ckpt": E["conc_ckpt"],
           "heldout_pool_idx": heldout, "strata": strata, "neighbours": nn,
           "composition": comp,
           "popularity": {"definition": "train users who can answer: ctx.cnt[i] for an item, "
                                        "answer_rate[c]*140768 for a concept",
                          "values": [float(x) for x in popv]},
           "canonical_snap": f0, "static_val_full": float(np.nanmean(sf)),
           "minutes": round((time.time() - t00) / 60, 2)}
    with open(SPLIT_JSON + ".tmp", "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    os.replace(SPLIT_JSON + ".tmp", SPLIT_JSON)
    log(f"[split] wrote {SPLIT_JSON}")
    return out


# ============================================================================ forced-opening rollout
def rollout_forced(E, pol, phi_t, pool, rows, forced, L=TOTAL_L, batch=1000):
    """`policy_net.rollout`, except the first `len(forced)` questions are IMPOSED.

    ★ A MECHANISM DIAGNOSTIC, declared as one -- it is not a pre-registered arm. The held-out draw
    took the tree's ROOT, so position 0 carries no supervision for any user and every later state is
    reached through an opening the policy was never taught. Imposing the root and letting the policy
    run from position 1 separates "the root was load-bearing" from "similarity failed everywhere".
    """
    from strategy_channel_suite import genv_latent, genv_eval_multi
    from ndcg_tree import branch_of_per_user
    n, P = len(rows), len(pool)
    asked = np.full((n, L), -1, np.int64)
    outs = np.full((n, L), -1, np.int8)
    asked_mask = np.zeros((n, P), bool)
    z = genv_latent(E, [], rows).numpy()
    phi_g = torch.from_numpy(phi_t)
    curves = {}
    pol.eval()
    for t in range(L):
        if t < len(forced):
            asked[:, t] = int(forced[t])
        else:
            hq = np.zeros((n, TOTAL_L - 1), np.int64); ho = np.zeros((n, TOTAL_L - 1), np.int64)
            hm = np.zeros((n, TOTAL_L - 1), np.float32)
            for j in range(t):
                hq[:, j] = asked[:, j]; ho[:, j] = outs[:, j]; hm[:, j] = 1.0
            pos = np.tile(np.asarray([[t, TOTAL_L - t]], np.float32), (n, 1))
            with torch.no_grad():
                for st in range(0, n, batch):
                    sl = slice(st, min(st + batch, n))
                    s = pol(torch.from_numpy(z[sl]), torch.from_numpy(phi_t[hq[sl]]),
                            torch.from_numpy(ho[sl]), torch.from_numpy(hm[sl]),
                            torch.from_numpy(pos[sl]), phi_g, torch.from_numpy(asked_mask[sl]))
                    asked[sl, t] = s.argmax(1).numpy()
        outs[:, t] = branch_of_per_user(E, rows, [pool[int(j)] for j in asked[:, t]])
        asked_mask[np.arange(n), asked[:, t]] = True
        need = (t + 1 == L)
        iseq = [[pool[int(j)][1] for j in asked[u, :t + 1] if pool[int(j)][0] == 0] for u in range(n)]
        cseq = [[pool[int(j)][1] for j in asked[u, :t + 1] if pool[int(j)][0] == 1] for u in range(n)]
        r = genv_eval_multi(E, rows, iseq, cseq, return_z=(t + 1 < L), score=need)
        if need:
            curves[L] = (r[0], r[1])
        if t + 1 < L:
            z = r[-1].numpy()
    assert (asked >= 0).all()
    return {"asked": asked, "outs": outs, "curves": curves}


# ============================================================================ the emission audit
def audit(args):
    """Emission at the held-out states, for every arm, plus the two comparators."""
    from strategy_channel_suite import (build_greedy_env, contract_selftest, genv_mean,
                                        CERT_TOWER, CERT_CONC)
    from ndcg_tree import build_pool, leak_assert
    t00 = time.time()
    sp = json.load(open(SPLIT_JSON))
    heldout = np.asarray(sp["heldout_pool_idx"], np.int64)
    hset = set(int(x) for x in heldout)
    nn = {int(k): [int(j) for j in v["pool_idx"]] for k, v in sp["neighbours"].items()}
    popv = np.asarray(sp["popularity"]["values"], np.float64)

    Dva = load_traj(train=False)
    pool = Dva["pool"]
    assert [[int(a), int(b)] for a, b in pool] == sp["pool"], "the split's pool is not this pool"
    lab_va = np.asarray(Dva["label_idx"], np.int64)
    P = len(pool)

    E = build_greedy_env(args.snapshot, "validation", args.concept_ckpt, "p1/audit")
    contract_selftest(E, E["rows"]); leak_assert(E, E["rows"], "p1/audit")
    rows = E["rows"]
    f0, _t = genv_mean(E, [], rows)
    assert round(f0, 8) == VAL_M0_FULL, f"CANONICAL SNAP FAILED: {f0!r}"
    log(f"[gate] canonical m=0 snap {f0!r} -- exact")
    pool_e, _i, _c = build_pool(E, 500, 500)
    assert pool_e == pool, "the env's pool is not the trajectory bank's pool"
    phi_t, norm = build_phi(E, pool)

    # the (user, position) states whose TRUE label is a held-out question
    U, T = np.nonzero(np.isin(lab_va, heldout))
    log(f"[states] {len(U):,} validation (user, position) states carry a held-out label")

    arms = {}
    for spec in args.ckpts.split(","):
        name, path = spec.split("=", 1)
        blob = torch.load(path, map_location="cpu")
        assert [(int(a), int(b)) for a, b in blob["pool"]] == pool, f"{name}: another pool"
        assert np.array_equal(blob["norm_mu"], norm[0]) and np.array_equal(blob["norm_sd"], norm[1]), \
            f"{name}: the checkpoint's descriptor standardisation differs from this env's"
        pol = Policy(E["ctx"].d, phi_t.shape[1], P, arch=blob["arch"], state=blob["state"],
                     use_pos=blob["use_pos"])
        pol.load_state_dict(blob["model"]); pol.eval()
        arms[name] = {"pol": pol, "ckpt": os.path.basename(path), "epoch": int(blob["epoch"]),
                      "val_full": float(blob["val_full"]), "val_tail": float(blob["val_tail"])}
        log(f"[arm] {name}: {os.path.basename(path)} epoch {blob['epoch']} "
            f"val_full {blob['val_full']:.6f}")

    phi_g = torch.from_numpy(phi_t)
    B = 2048
    res = {}
    for name, a in arms.items():
        top1 = np.zeros(len(U), np.int64); in5 = np.zeros(len(U), bool)
        nbr = np.zeros(len(U), bool); rank = np.zeros(len(U), np.int64)
        with torch.no_grad():
            for st in range(0, len(U), B):
                sl = slice(st, min(st + B, len(U)))
                u, t = U[sl], T[sl]
                z, hp, ho, hm, pos, am, y = make_batch(Dva, phi_t, u, t, P)
                s = a["pol"](z, hp, ho, hm, pos, phi_g, am)
                s = s.numpy()
                top5 = np.argsort(-s, axis=1)[:, :5]
                yv = y.numpy()
                top1[sl] = top5[:, 0]
                in5[sl] = (top5 == yv[:, None]).any(1)
                rank[sl] = (s > s[np.arange(len(yv)), yv][:, None]).sum(1)
                nbr[sl] = [int(top5[i, 0]) in nn[int(yv[i])] for i in range(len(yv))]
        yv_all = lab_va[U, T]
        hit1 = (top1 == yv_all)
        per_q = {}
        for q in sorted(hset):
            m = yv_all == q
            if not m.any():
                continue
            per_q[str(q)] = {"channel": "item" if pool[q][0] == 0 else "concept",
                             "id": int(pool[q][1]), "n_states": int(m.sum()),
                             "top1": float(hit1[m].mean()), "top5": float(in5[m].mean()),
                             "neighbour_credit": float(nbr[m].mean()),
                             "median_rank_of_the_held_out_question": float(np.median(rank[m]))}
        res[name] = {"ckpt": a["ckpt"], "epoch": a["epoch"], "val_full": a["val_full"],
                     "val_tail": a["val_tail"], "n_states": int(len(U)),
                     "top1_emission": float(hit1.mean()), "top5_emission": float(in5.mean()),
                     "neighbour_credit_top1_in_NN5": float(nbr.mean()),
                     "neighbour_credit_or_top1": float((nbr | hit1).mean()),
                     "median_rank_of_the_held_out_question": float(np.median(rank)),
                     "mean_rank_of_the_held_out_question": float(rank.mean()),
                     "frac_emitting_ANY_held_out_question_top1":
                         float(np.isin(top1, heldout).mean()),
                     "per_question": per_q}
        log(f"[{name}] top1 {res[name]['top1_emission']:.4f} top5 {res[name]['top5_emission']:.4f} "
            f"neighbour {res[name]['neighbour_credit_top1_in_NN5']:.4f} "
            f"median rank {res[name]['median_rank_of_the_held_out_question']:.0f}")

    # ---- the dumb comparator: a fixed popularity ranking ------------------------------------------
    pop_order = np.argsort(-popv)
    pop_rank_of = np.empty(P, np.int64)
    pop_rank_of[pop_order] = np.arange(P)
    top1p = np.zeros(len(U), np.int64); in5p = np.zeros(len(U), bool)
    nbrp = np.zeros(len(U), bool)
    for st in range(0, len(U), B):
        sl = slice(st, min(st + B, len(U)))
        u, t = U[sl], T[sl]
        _z, _hp, _ho, _hm, _pos, am, y = make_batch(Dva, phi_t, u, t, P)
        s = -pop_rank_of.astype(np.float64)[None, :].repeat(len(u), 0)
        s[am.numpy()] = -np.inf
        top5 = np.argsort(-s, axis=1)[:, :5]
        yv = y.numpy()
        top1p[sl] = top5[:, 0]
        in5p[sl] = (top5 == yv[:, None]).any(1)
        nbrp[sl] = [int(top5[i, 0]) in nn[int(yv[i])] for i in range(len(yv))]
    yv_all = lab_va[U, T]
    res["popularity_emitter"] = {
        "what": sp["popularity"]["definition"], "n_states": int(len(U)),
        "top1_emission": float((top1p == yv_all).mean()), "top5_emission": float(in5p.mean()),
        "neighbour_credit_top1_in_NN5": float(nbrp.mean())}
    log(f"[popularity_emitter] top1 {res['popularity_emitter']['top1_emission']:.4f} "
        f"top5 {res['popularity_emitter']['top5_emission']:.4f}")

    # ---- the mechanism diagnostic: impose the held-out opening ------------------------------------
    forced_diag = None
    if args.forced_arms:
        from strategy_channel_suite import genv_eval
        static_seq = [(int(a), int(b)) for a, b in json.load(open(STATIC_JSON))["sequence"]]
        sfv, stv = genv_eval(E, static_seq, rows)
        assert abs(float(np.nanmean(sfv)) - STATIC_VAL_FULL) < 1e-12, "BANKED-HEADLINE ASSERT FAILED"
        root = int(lab_va[0, 0])
        assert (lab_va[:, 0] == root).all(), "the tree's opening is not one question for every user"
        forced_diag = {"forced_first_question_pool_idx": root,
                       "forced_is_held_out": bool(root in hset),
                       "channel": "item" if pool[root][0] == 0 else "concept",
                       "id": int(pool[root][1]),
                       "why": "the held-out draw took the tree's ROOT, so position 0 carries no "
                              "supervision for any user. Imposing it separates 'the root was "
                              "load-bearing' from 'similarity failed everywhere'. A POST-HOC "
                              "mechanism diagnostic, not a pre-registered arm.",
                       "arms": {}}
        for name in args.forced_arms.split(","):
            if name not in arms:
                continue
            ro = rollout_forced(E, arms[name]["pol"], phi_t, pool, rows, [root])
            f, t = ro["curves"][TOTAL_L]
            forced_diag["arms"][name] = {
                "val_full_with_the_root_imposed": float(np.nanmean(f)),
                "val_tail_with_the_root_imposed": float(np.nanmean(t)),
                "val_full_free": arms[name]["val_full"],
                "recovered_of_the_gap_to_E1": float(
                    (float(np.nanmean(f)) - arms[name]["val_full"]) /
                    (PN.E1_VAL_FULL - arms[name]["val_full"]))
                if arms[name]["val_full"] < PN.E1_VAL_FULL else None,
                "vs_E1_paired": paired_bootstrap(f - np.asarray(
                    np.load(PN.E1_PER_USER)["policy_full"], np.float64)),
                "vs_static_paired": paired_bootstrap(f - sfv),
                "n_distinct_interviews": len({tuple(r) for r in ro["asked"]}),
                "agreement_with_the_tree_after_the_forced_opening": float(
                    (ro["asked"][:, 1:] == lab_va[:, 1:]).mean())}
            log(f"[forced {name}] val full {float(np.nanmean(f)):.5f} (free "
                f"{arms[name]['val_full']:.5f}, E1 {PN.E1_VAL_FULL:.5f}); agreement with the tree "
                f"after the forced opening "
                f"{forced_diag['arms'][name]['agreement_with_the_tree_after_the_forced_opening']:.4f}")

    out = {"analysis": "policy_p1_audit", "design_sheet":
           "docs/design/PAPER_B_POLICY_DESIGN.md I.1", "citable": True, "test_touched": False,
           "split": "validation", "n_held_out_questions": int(len(heldout)),
           "composition": sp["composition"], "arms": res,
           "forced_opening_diagnostic": forced_diag,
           "minutes": round((time.time() - t00) / 60, 2)}
    p = os.path.join(P1_DIR, "p1_audit.json")
    with open(p + ".tmp", "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    os.replace(p + ".tmp", p)
    log(f"[audit] wrote {p}")
    return out


def main():
    ap = argparse.ArgumentParser(description="P1 (Part I.1): generalisation across questions")
    ap.add_argument("--mode", required=True, choices=["split", "audit"])
    ap.add_argument("--ckpts", default="", help="name=path,name=path -- the arms to audit")
    ap.add_argument("--forced_arms", default="",
                    help="comma-separated arm names to also run with the tree's opening IMPOSED "
                         "(a post-hoc mechanism diagnostic, declared as one)")
    ap.add_argument("--snapshot", default=None)
    ap.add_argument("--concept_ckpt", default=None)
    ap.add_argument("--threads", type=int, default=None, help="consumed by strategy_channel_suite")
    ap.add_argument("--full_threads", action="store_true")
    ap.add_argument("--i_have_author_goahead", action="store_true")
    args = ap.parse_args()
    assert args.i_have_author_goahead, "REFUSING TO RUN without --i_have_author_goahead"
    from strategy_channel_suite import CERT_TOWER, CERT_CONC
    args.snapshot = args.snapshot or CERT_TOWER
    args.concept_ckpt = args.concept_ckpt or CERT_CONC
    log("#" * 110)
    log("# P1 -- GENERALISATION ACROSS QUESTIONS (design sheet Part I.1). VALIDATION ONLY.")
    log("# The 10,000 canonical TEST users are NOT built, NOT loaded, NOT touched.")
    log("#" * 110)
    return build_split(args) if args.mode == "split" else audit(args)


if __name__ == "__main__":
    main()
