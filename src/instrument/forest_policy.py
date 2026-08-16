r"""EXPERIMENT L2a -- THE TWO-TEACHER FOREST: one policy, two cohort-measured teachers.

Design sheet: `docs/design/PAPER_B_POLICY_DESIGN.md` **PART L.2 is binding**; **PART K.6** says why a
per-user label is unlearnable and must never come back. Committed before the run (HARD RULE 10).
VALIDATION ONLY -- `--split test` does not exist in this parser, the 10,000 canonical TEST users are
never built, never loaded, never named, and `test_touched=false` is stamped on every artifact.

THE DESIGN LAW (the author's, 2026-08-17): DENOISING IS COHORT MEASUREMENT
--------------------------------------------------------------------------
A conditioning grouping is admissible iff (a) its signature is OBSERVABLE AT ASK-TIME -- otherwise
the label is clairvoyant and unlearnable (K.6) -- and (b) its groups are LARGE ENOUGH TO MEASURE
HONESTLY (>= ~500 users; below 200 is the -0.724 noise regime). Both teachers here satisfy both:

  T0  THE COLD TREE. Signature: nothing volunteered. Label: the tree's next question at the state,
      a function of the asked-and-answered history alone. Banked as E1's trajectory bank over the
      140,768 TRAIN users. Prior: +0.02794 validation / +0.01035 test.
  T1  THE STATED-GENRE ORDERS. Signature: "my favourite genre is G" -- volunteered, observable, and
      encoded as an answered question per L.1 (history token + fold + mask). Label: the G-group's
      own greedy item order at that depth, measured on the cohort by `group_greedy_genre.py` and
      banked in `groupgreedy_genre15.json`. Only groups at or above `--min_group` (declared 500,
      the design law's floor) become teachers; the rest are NOT taught and fall back to the tree.

  ✗ inferred-genre-of-items (V: null in every big group) and ✗ per-movie (n=1, the clairvoyant trap)
    are DEAD by measurement and are not implemented here, deliberately.

WHAT IS HONESTLY EXPECTED, DECLARED BEFORE THE RUN
---------------------------------------------------
T1's orders are ITEM-ONLY (the genre question was free in that experiment, so the group searched 15
items). The tree spends most of its budget on CONCEPTS. So T1 is a teacher over a poorer action
space, and at genre-seeded states the tree may well SCORE better than T1 does. That does not make
the forest wrong -- the forest's claim is *exact where a teacher is defined* -- but it does mean the
score table must show policy, T1's own order AND the tree side by side at those states, and it does.

★ THE FALSIFIER: INTERFERENCE. Fidelity to T0 and fidelity to T1 are reported TOGETHER, at every
checkpoint, always. If learning T1 costs T0, the single-policy claim fails at two teachers and
scaling the forest is dead -- reported dead, not retried.

THE FIREWALL (K.2, unchanged)
------------------------------
The 10,000 validation users split 8,000 FIT / 2,000 HOLDOUT by user, seed 98765, recomputed here
from `pu_labels.fit_holdout_split` rather than read from a file. T1 states are trained on FIT users
only; every gate is proven on HOLDOUT. T0's replay is the TRAIN cohort, disjoint from both.
DECLARED, because it is a real limitation: the genre orders were SEARCHED on all 10,000 validation
users, so HOLDOUT is held out from the POLICY, not from the TEACHER. The behavioural gate compares
the policy against that same teacher, so both sides carry the teacher's selection equally; no
comparison in this file treats a group order as if it were held-out-clean.

USAGE
  python src/instrument/forest_policy.py --mode composition --i_have_author_goahead
  python src/instrument/forest_policy.py --mode states --i_have_author_goahead
  python src/instrument/forest_policy.py --mode fit --arm main --i_have_author_goahead
  python src/instrument/forest_policy.py --mode fit --arm shuffle --i_have_author_goahead
  python src/instrument/forest_policy.py --mode evaluate --arm main --i_have_author_goahead
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
import torch
import torch.nn as nn

import policy_net as PN
from policy_net import (OUTDIR, SEED, TOTAL_L, STATIC_JSON, STATIC_VAL_FULL, VAL_M0_FULL,
                        E1_VAL_FULL, E1_PER_USER, Policy, build_phi, paired_bootstrap,
                        load_traj, make_batch, rollout as unseeded_rollout, log)

FOREST_DIR = os.path.join(OUTDIR, "forest")
E1_CKPT = os.path.join(OUTDIR, "policy_main_equivariant_both.pt")
GENRE_JSON = os.path.join(OUTDIR, "groupgreedy_genre15.json")
GENRE_VAL_POLICYB = os.path.join(OUTDIR, "groupgreedy_genre15_VAL_policyB.json")
STATE_NPZ = os.path.join(FOREST_DIR, "forest_states.npz")
STATE_JSON = os.path.join(FOREST_DIR, "forest_states.json")

MIN_GROUP = 500            # the design law's cohort floor. DECLARED, not tuned.
GENRE_TOKEN_OUTCOME = "measured"   # the stated genre enters with the user's OWN ternary answer,
                                   # exactly as every other question in this codebase does.
SHUFFLE_SEED = 98771       # declared; the teacher-assignment permutation for the control arm


# ============================================================ the stated-genre concept, by name
def genre_to_concept(ctx, genres):
    """MovieLens genre -> the genome-tag concept id that names it.

    The concept vocabulary IS the genome tag vocabulary (`ctx.tags` holds tagIds), so a stated genre
    has a literal concept: "action" the tag is the question "do you like action films?". The match is
    on the tag STRING, lower-cased, with the two punctuation variants MovieLens uses spelled out. A
    genre with no tag, or whose tag is outside the question pool, cannot be a teacher signature and
    is dropped with the reason printed -- never silently.
    """
    path = os.path.join(_ROOT, "data", "movielens", "genome-tags.csv")
    assert os.path.exists(path), f"genome tag names missing: {path}"
    name = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            name[int(r["tagId"])] = r["tag"].strip().lower()
    by_name = {}
    for cid, t in enumerate(list(ctx.tags)):
        s = name.get(int(t)) if str(t).isdigit() or isinstance(t, (int, np.integer)) else str(t)
        if s is not None:
            by_name.setdefault(str(s).lower(), cid)
    out, missing = {}, []
    for g in genres:
        for cand in (g.lower(), g.lower().replace("-", " "), g.lower().replace("-", "")):
            if cand in by_name:
                out[g] = int(by_name[cand]); break
        else:
            missing.append(g)
    return out, missing, len(by_name)


# ============================================================ the banked group orders
def load_group_orders():
    b = json.load(open(GENRE_JSON))
    assert b["test_touched"] is False and b["split"] == "validation"
    assert b["tower"] == "t2i26c_ep31.pt" and b["concept_ckpt"] == "concept_cv3_ep08.pt", \
        "the banked genre orders are not on the certified pair"
    orders = {g: [int(i) for i in v["order"]] for g, v in b["groups"].items()}
    sizes = {g: int(v["n_users"]) for g, v in b["groups"].items()}
    return orders, sizes, b


# ============================================================ state generation (mode: states)
def build_states(E, rows, pool, teachers, orders, gmap, groups, args):
    """Roll each teacher's own order forward over its own users, banking the state at every depth.

    For a user whose stated genre is G, the state at depth j is: the G concept answered (history slot
    0, the user's own ternary outcome, folded, masked) plus the first j items of G's order, each with
    the user's own ternary outcome. The LABEL at that state is G's order[j] -- the group's measured
    next question, identical for every user in the group at that depth. That is the point: it is a
    COHORT measurement, not a per-user argmax (K.6).

    Budget accounting: the genre statement occupies one of the 16 turns, and the order is 15 items,
    so a G-user's full interview is exactly 16 evidence questions -- the same budget as every other
    arm in this chapter.
    """
    from strategy_channel_suite import genv_eval_multi
    from ndcg_tree import branch_of_per_user
    qidx = {q: j for j, q in enumerate(pool)}
    Z, HQ, HO, HM, LAB, URW, TCH, DEP = [], [], [], [], [], [], [], []
    per_teacher = {}
    for ti, g in enumerate(teachers):
        grows = np.asarray(sorted(groups[g]), np.int64)
        cq = (1, int(gmap[g]))
        seq = [(0, int(i)) for i in orders[g]]
        assert all(q in qidx for q in seq), f"{g}'s order contains an item outside the pool"
        n = len(grows)
        gout = branch_of_per_user(E, grows, [cq] * n)
        hq = np.zeros((n, TOTAL_L - 1), np.int64); ho = np.zeros((n, TOTAL_L - 1), np.int64)
        hm = np.zeros((n, TOTAL_L - 1), np.float32)
        hq[:, 0] = qidx[cq]; ho[:, 0] = gout; hm[:, 0] = 1.0
        iseq = [[] for _ in range(n)]
        cseq = [[int(gmap[g])] for _ in range(n)]
        z = genv_eval_multi(E, grows, iseq, cseq, score=False, return_z=True)[-1].numpy()
        for j in range(len(seq)):
            Z.append(z.astype(np.float32)); HQ.append(hq.copy()); HO.append(ho.copy())
            HM.append(hm.copy()); LAB.append(np.full(n, qidx[seq[j]], np.int64))
            URW.append(grows.copy()); TCH.append(np.full(n, ti, np.int8))
            DEP.append(np.full(n, j, np.int8))
            oo = branch_of_per_user(E, grows, [seq[j]] * n)
            hq[:, j + 1] = qidx[seq[j]]; ho[:, j + 1] = oo; hm[:, j + 1] = 1.0
            for u in range(n):
                iseq[u].append(int(seq[j][1]))
            if j + 1 < len(seq):
                z = genv_eval_multi(E, grows, iseq, cseq, score=False,
                                    return_z=True)[-1].numpy()
            log(f"    [states {g}] depth {j + 1}/{len(seq)} done")
        per_teacher[g] = {"n_users": int(n), "concept_id": int(gmap[g]),
                          "genre_token_outcome_mix": {k: float((np.asarray(gout) == v).mean())
                                                      for k, v in PN.OUTCOME.items()},
                          "order_pool_idx": [int(qidx[q]) for q in seq],
                          "n_states": int(n * len(seq))}
    S = {"z": np.concatenate(Z), "hq": np.concatenate(HQ), "ho": np.concatenate(HO),
         "hm": np.concatenate(HM), "label": np.concatenate(LAB), "row": np.concatenate(URW),
         "teacher": np.concatenate(TCH), "depth": np.concatenate(DEP)}
    return S, per_teacher


def forest_batch(S, phi_t, idx, P, teacher_of_row=None, relabel=None):
    """One hard-label batch from the banked T1 states. The position feature counts the genre
    statement, because it WAS a turn: at depth j the state is `t = j + 1`, `16 - t` remaining."""
    hq = np.asarray(S["hq"][idx], np.int64); ho = np.asarray(S["ho"][idx], np.int64)
    hm = np.asarray(S["hm"][idx], np.float32)
    B, T = hq.shape
    am = np.zeros((B, P), bool)
    am[np.repeat(np.arange(B), T)[hm.ravel() > 0], hq.ravel()[hm.ravel() > 0]] = True
    y = np.asarray(S["label"][idx], np.int64) if relabel is None else relabel
    am[np.arange(B), y] = False
    pos_t = (np.asarray(S["depth"][idx], np.float32) + 1.0)
    pos = np.stack([pos_t, TOTAL_L - pos_t], 1)
    return (torch.from_numpy(np.ascontiguousarray(S["z"][idx]).astype(np.float32)),
            torch.from_numpy(phi_t[hq]), torch.from_numpy(ho), torch.from_numpy(hm),
            torch.from_numpy(pos), torch.from_numpy(am), torch.from_numpy(y))


# ============================================================ fidelity read-outs (the falsifier)
def fidelity_T0(pol, phi_t, D, P, pos_idx, batch=2048):
    """Top-1 against the TREE's own next question, at the tree's own states, on `pos_idx` users of
    the validation trajectory bank. E1 scores 1.0000 here; this is the number T1 may not cost."""
    phi_g = torch.from_numpy(phi_t)
    pol.eval()
    hit = tot = 0
    with torch.no_grad():
        for t in range(TOTAL_L):
            for st in range(0, len(pos_idx), batch):
                u = np.asarray(pos_idx[st:st + batch], np.int64)
                tt = np.full(len(u), t, np.int64)
                z, hp, ho, hm, pos, am, y = make_batch(D, phi_t, u, tt, P)
                s = pol(z, hp, ho, hm, pos, phi_g, am)
                hit += int((s.argmax(1) == y).sum()); tot += len(u)
    return hit / tot


def fidelity_T1(pol, phi_t, S, P, idx, batch=2048):
    """Top-1 against the matching GROUP ORDER's next question, at genre-signature states."""
    phi_g = torch.from_numpy(phi_t)
    pol.eval()
    hit = tot = 0
    per = {}
    with torch.no_grad():
        for st in range(0, len(idx), batch):
            ix = idx[st:st + batch]
            z, hp, ho, hm, pos, am, y = forest_batch(S, phi_t, ix, P)
            s = pol(z, hp, ho, hm, pos, phi_g, am)
            ok = (s.argmax(1) == y).numpy()
            hit += int(ok.sum()); tot += len(ix)
            for ti in np.unique(S["teacher"][ix]):
                m = np.asarray(S["teacher"][ix] == ti)
                a, b = per.get(int(ti), (0, 0))
                per[int(ti)] = (a + int(ok[m].sum()), b + int(m.sum()))
    return hit / tot, {int(k): v[0] / v[1] for k, v in per.items()}


# ============================================================ shared env + gates
def open_env(tag):
    from strategy_channel_suite import (build_greedy_env, contract_selftest, genv_eval, genv_mean,
                                        CERT_TOWER, CERT_CONC)
    from ndcg_tree import build_pool, leak_assert, branch_of_selftest
    E = build_greedy_env(CERT_TOWER, "validation", CERT_CONC, tag)
    contract_selftest(E, E["rows"])
    leak_assert(E, E["rows"], f"{tag}/val")
    rows = E["rows"]
    assert len(rows) == 10000
    f0, t0_ = genv_mean(E, [], rows)
    assert round(f0, 8) == VAL_M0_FULL, f"CANONICAL SNAP FAILED: {f0!r} != {VAL_M0_FULL}"
    log(f"[gate] canonical m=0 snap full={f0!r} tail={t0_!r} -- exact")
    D = load_traj(train=False)
    pool = D["pool"]
    pool_e, _i, _c = build_pool(E, 500, 500)
    assert pool_e == pool, "the env's pool is not the trajectory bank's pool"
    assert list(np.asarray(D["uid"], np.int64)) == list(np.asarray(rows, np.int64)), \
        "the validation trajectory rows are not the env's rows, in order"
    branch_of_selftest(E, rows[:2000], [pool[0], pool[500], (1, 114)])
    static_seq = [(int(a), int(b)) for a, b in json.load(open(STATIC_JSON))["sequence"]]
    sf, _ = genv_eval(E, static_seq, rows)
    assert abs(float(np.nanmean(sf)) - STATIC_VAL_FULL) < 1e-12, "BANKED-HEADLINE ASSERT FAILED"
    log(f"[gate] greedy_static_policyB_combined reproduces EXACTLY: {float(np.nanmean(sf))!r}")
    return E, rows, D, pool


def firewall(rows):
    import pu_labels as PL
    fit, hold, info = PL.fit_holdout_split(np.asarray(rows, np.int64))
    log(f"[firewall] {len(fit):,} FIT / {len(hold):,} HOLDOUT by user, seed {info['seed']}; "
        f"fit sha1 {info['fit_sha1'][:12]} holdout sha1 {info['holdout_sha1'][:12]}")
    return fit, hold, info


def teacher_setup(E, rows, pool, args):
    """The teachers, their signatures and their cohorts -- PRINTED BEFORE ANY TIME IS SPENT."""
    from group_greedy_genre import load_item_genres, partition_by_genre
    orders, banked_sizes, gblob = load_group_orders()
    gen, vocab = load_item_genres(E["ctx"].ni)
    groups, n_no_liked, n_ungenred = partition_by_genre(E, rows, gen)
    sizes = {g: len(v) for g, v in groups.items()}
    for g, n in banked_sizes.items():
        assert sizes.get(g, 0) == n, (
            f"the genre partition does not reproduce the banked cohort: {g} is {sizes.get(g, 0)} "
            f"here and {n} in groupgreedy_genre15.json")
    log(f"[chain] the genre partition reproduces all {len(banked_sizes)} banked group sizes exactly")
    gmap, missing, n_named = genre_to_concept(E["ctx"], sorted(sizes))
    inpool = {g: c for g, c in gmap.items() if (1, c) in set(pool)}
    teachers = [g for g in sorted(sizes, key=lambda x: -sizes[x])
                if sizes[g] >= args.min_group and g in inpool and g in orders]
    dropped = {g: {"n_users": sizes[g],
                   "reason": ("below the cohort floor" if sizes[g] < args.min_group else
                              "no genome tag names this genre" if g not in gmap else
                              "its concept is outside the 500-question pool" if g not in inpool else
                              "no banked group order")}
               for g in sorted(sizes, key=lambda x: -sizes[x]) if g not in teachers}
    info = {"min_group": args.min_group, "n_named_tags": n_named,
            "genres_with_no_tag": missing,
            "teachers": [{"genre": g, "n_users": sizes[g], "concept_id": inpool[g],
                          "order_len": len(orders[g])} for g in teachers],
            "n_users_covered": int(sum(sizes[g] for g in teachers)),
            "frac_users_covered": float(sum(sizes[g] for g in teachers) / len(rows)),
            "not_taught": dropped,
            "n_users_no_liked_foldin": n_no_liked, "n_users_liked_but_ungenred": n_ungenred}
    log("[composition] " + json.dumps(info, default=float))
    return teachers, orders, inpool, groups, info


# ============================================================ modes
def mode_composition(args):
    E, rows, D, pool = open_env("forest/comp")
    teachers, orders, gmap, groups, info = teacher_setup(E, rows, pool, args)
    fit, hold, fw = firewall(rows)
    fs, hs = set(fit.tolist()), set(hold.tolist())
    info["by_teacher_split"] = {g: {"fit": int(sum(1 for r in groups[g] if r in fs)),
                                    "holdout": int(sum(1 for r in groups[g] if r in hs)),
                                    "states_fit": int(sum(1 for r in groups[g] if r in fs)
                                                      * len(orders[g]))}
                                for g in teachers}
    info["firewall"] = fw
    os.makedirs(FOREST_DIR, exist_ok=True)
    p = os.path.join(FOREST_DIR, "forest_composition.json")
    with open(p, "w") as fh:
        json.dump({"analysis": "forest_composition", "design_sheet":
                   "docs/design/PAPER_B_POLICY_DESIGN.md L.2", "test_touched": False,
                   "split": "validation", **info}, fh, indent=2, default=float)
    log(f"[composition] wrote {p}")
    log("[composition] " + json.dumps(info["by_teacher_split"]))


def mode_states(args):
    assert not os.path.exists(STATE_NPZ) or args.overwrite, (
        f"{STATE_NPZ} exists. Pass --overwrite only if you mean to replace a banked artifact.")
    t0 = time.time()
    E, rows, D, pool = open_env("forest/states")
    teachers, orders, gmap, groups, info = teacher_setup(E, rows, pool, args)
    assert teachers, "no teacher group clears the cohort floor -- nothing to distil"
    S, per_teacher = build_states(E, rows, pool, teachers, orders, gmap, groups, args)
    os.makedirs(FOREST_DIR, exist_ok=True)
    np.savez_compressed(STATE_NPZ, **S)
    blob = {"analysis": "forest_states", "design_sheet":
            "docs/design/PAPER_B_POLICY_DESIGN.md L.2", "test_touched": False,
            "split": "validation", "teachers": teachers, "composition": info,
            "per_teacher": per_teacher, "n_states": int(len(S["label"])),
            "genre_token_outcome": GENRE_TOKEN_OUTCOME,
            "pool_contract": E["pool_contract"], "tower": E["tower"], "conc_ckpt": E["conc_ckpt"],
            "minutes": round((time.time() - t0) / 60, 2)}
    with open(STATE_JSON, "w") as fh:
        json.dump(blob, fh, indent=2, default=float)
    log(f"[states] banked {len(S['label']):,} T1 states over {len(teachers)} teachers "
        f"({blob['minutes']:.1f}m)")


def load_states():
    z = np.load(STATE_NPZ)
    S = {k: z[k] for k in z.files}
    blob = json.load(open(STATE_JSON))
    return S, blob


def mode_fit(args):
    assert args.i_have_author_goahead, "REFUSING TO FIT without --i_have_author_goahead"
    torch.manual_seed(SEED); np.random.seed(SEED)
    rng = np.random.default_rng(SEED)
    t00 = time.time()
    S, sblob = load_states()
    teachers = sblob["teachers"]
    E, rows, Dva, pool = open_env(f"forest/fit_{args.arm}")
    P = len(pool)
    phi_t, norm = build_phi(E, pool)
    phi_g = torch.from_numpy(phi_t)
    fit, hold, fw = firewall(rows)

    # ---- the firewall, enforced on the STATES, not just declared -------------------------------
    fs = set(int(x) for x in fit.tolist()); hs = set(int(x) for x in hold.tolist())
    in_fit = np.asarray([int(r) in fs for r in S["row"]])
    in_hold = np.asarray([int(r) in hs for r in S["row"]])
    assert (in_fit | in_hold).all(), "a banked state belongs to no side of the firewall"
    fit_ix = np.flatnonzero(in_fit); hold_ix = np.flatnonzero(in_hold)
    log(f"[firewall] T1 states: {len(fit_ix):,} FIT (trainable) / {len(hold_ix):,} HOLDOUT "
        f"(never trained on, never selected on)")

    # ---- the shuffle control: teacher ASSIGNMENTS permuted across signatures --------------------
    relabel_map = None
    if args.arm == "shuffle":
        perm = np.random.default_rng(SHUFFLE_SEED).permutation(len(teachers))
        while any(perm == np.arange(len(teachers))):
            perm = np.random.default_rng(SHUFFLE_SEED + int(perm.sum())).permutation(len(teachers))
        ords = [sblob["per_teacher"][g]["order_pool_idx"] for g in teachers]
        relabel_map = {ti: ords[int(perm[ti])] for ti in range(len(teachers))}
        log(f"[arm] SHUFFLE CONTROL: teacher assignments permuted across signatures "
            f"{ {teachers[i]: teachers[int(perm[i])] for i in range(len(teachers))} }. Every genre "
            f"still gets a measured group order -- just not its own. An arm that matches `main` has "
            f"learned the orders, not the conditioning.")

    def labels_for(ix):
        if relabel_map is None:
            return None
        return np.asarray([relabel_map[int(S["teacher"][i])][int(S["depth"][i])] for i in ix],
                          np.int64)

    # ---- the model: E1, warm-started and proved ------------------------------------------------
    import pu_policy as PU
    pol = Policy(E["ctx"].d, phi_t.shape[1], P, arch="equivariant", state="both", use_pos=True)
    e1 = PU.warm_start(pol, E1_CKPT, pool, norm)

    # ---- the replay: T0's own supervision, unchanged --------------------------------------------
    Dtr = load_traj(train=True)
    assert [(int(a), int(b)) for a, b in Dtr["pool"]] == pool, "the replay pool is not this pool"
    n_rep = len(Dtr["uid"]) * TOTAL_L
    n_t1_b = int(round(args.batch * args.t1_frac))
    n_t0_b = args.batch - n_t1_b
    steps = int(np.ceil(len(fit_ix) / max(n_t1_b, 1)))
    log(f"[mixture] batch {args.batch} = {n_t1_b} T1 (stated-genre) + {n_t0_b} T0 (tree replay); "
        f"t1_frac {args.t1_frac} DECLARED, not tuned post hoc; one epoch = one shuffled pass over "
        f"the {len(fit_ix):,} FIT T1 states = {steps} steps")

    # a FIT-dev slice for selection, by user, never HOLDOUT
    dev_rng = np.random.default_rng(98767)
    dev_users = set(int(x) for x in
                    dev_rng.permutation(fit)[:int(round(0.10 * len(fit)))].tolist())
    dev_ix = np.asarray([i for i in fit_ix if int(S["row"][i]) in dev_users], np.int64)
    tr_ix = np.asarray([i for i in fit_ix if int(S["row"][i]) not in dev_users], np.int64)
    val_pos = {int(r): j for j, r in enumerate(np.asarray(Dva["uid"], np.int64))}
    dev_pos = np.asarray(sorted(val_pos[u] for u in dev_users), np.int64)
    log(f"[select] {len(tr_ix):,} T1 train states / {len(dev_ix):,} T1 dev states "
        f"({len(dev_users):,} FIT-dev users). Selection NEVER sees HOLDOUT.")

    opt = torch.optim.Adam(pol.parameters(), lr=args.lr)
    lossf = nn.CrossEntropyLoss()
    stem = f"forest_{args.arm}" + (f"_{args.tag}" if args.tag else "")
    ckpt = os.path.join(FOREST_DIR, stem + ".pt")
    for suffix in (".pt", "_history.json"):
        p_ = os.path.join(FOREST_DIR, stem + suffix)
        assert not os.path.exists(p_) or args.overwrite, (
            f"{os.path.basename(p_)} exists. Use --tag, or --overwrite deliberately (HARD RULE 11e).")

    # ★ the falsifier, measured BEFORE any gradient step, so "it moved" is readable
    f0_T0 = fidelity_T0(pol, phi_t, Dva, P, dev_pos)
    f0_T1, f0_T1_per = fidelity_T1(pol, phi_t, S, P, dev_ix)
    log(f"[epoch 0 = E1] T0 fidelity {f0_T0:.4f} | T1 fidelity {f0_T1:.4f} "
        f"per-teacher { {teachers[k]: round(v, 4) for k, v in f0_T1_per.items()} }")

    hist = [{"epoch": 0, "T0_fidelity": f0_T0, "T1_fidelity": f0_T1,
             "T1_per_teacher": {teachers[k]: v for k, v in f0_T1_per.items()}, "note": "E1, no step"}]
    best = None
    for ep in range(1, args.epochs + 1):
        pol.train()
        perm = tr_ix[rng.permutation(len(tr_ix))]
        tot = seen = hit = 0.0
        te = time.time()
        nst = int(np.ceil(len(perm) / max(n_t1_b, 1)))
        for si in range(nst):
            ix = perm[si * n_t1_b:(si + 1) * n_t1_b]
            if not len(ix):
                continue
            z, hp, ho, hm, pos, am, y = forest_batch(S, phi_t, ix, P, relabel=labels_for(ix))
            e = rng.integers(0, n_rep, size=n_t0_b)
            u = (e // TOTAL_L).astype(np.int64); t = (e % TOTAL_L).astype(np.int64)
            z2, hp2, ho2, hm2, pos2, am2, y2 = make_batch(Dtr, phi_t, u, t, P)
            s1 = pol(z, hp, ho, hm, pos, phi_g, am)
            s2 = pol(z2, hp2, ho2, hm2, pos2, phi_g, am2)
            loss = (lossf(s1, y) * len(ix) + lossf(s2, y2) * len(u)) / (len(ix) + len(u))
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * len(ix); seen += len(ix)
            hit += int((s1.argmax(1) == y).sum())
            if (si + 1) % 200 == 0:
                log(f"[ep{ep} {si + 1}/{nst}] loss {tot / seen:.4f} T1-top1 {hit / seen:.4f} "
                    f"({(time.time() - te) / 60:.1f}m)")
        fT0 = fidelity_T0(pol, phi_t, Dva, P, dev_pos)
        fT1, fT1p = fidelity_T1(pol, phi_t, S, P, dev_ix)
        rec = {"epoch": ep, "train_loss": tot / seen, "train_T1_top1": hit / seen,
               "T0_fidelity": fT0, "T1_fidelity": fT1,
               "T1_per_teacher": {teachers[k]: v for k, v in fT1p.items()},
               "minutes": round((time.time() - te) / 60, 2)}
        hist.append(rec)
        log(f"[ep{ep}] loss {rec['train_loss']:.4f} | ★ T0 fidelity {fT0:.4f} (E1: {f0_T0:.4f}) | "
            f"★ T1 fidelity {fT1:.4f} (E1: {f0_T1:.4f}) | {rec['minutes']:.1f}m")
        # SELECTION: the sum of the two fidelities on FIT-dev. Declared. It cannot prefer an arm that
        # buys T1 by giving up T0 -- which is exactly the failure mode under test.
        sc = fT0 + fT1
        if best is None or sc > best["score"]:
            best = {"score": sc, "epoch": ep, "rec": rec}
            torch.save({"model": pol.state_dict(), "arch": "equivariant", "state": "both",
                        "use_pos": True, "d_phi": int(phi_t.shape[1]), "n_pool": P, "pool": pool,
                        "norm_mu": norm[0], "norm_sd": norm[1], "epoch": ep,
                        "val_full": float("nan"), "arm": args.arm, "teachers": teachers,
                        "t1_frac": args.t1_frac, "lr": args.lr,
                        "selection": "T0_fidelity + T1_fidelity on FIT-dev"}, ckpt)
            log(f"  [ckpt] epoch {ep} saved (selection score {sc:.4f})")
    with open(os.path.join(FOREST_DIR, stem + "_history.json"), "w") as fh:
        json.dump({"analysis": "forest_fit", "arm": args.arm, "test_touched": False,
                   "split": "validation", "teachers": teachers, "t1_frac": args.t1_frac,
                   "lr": args.lr, "batch": args.batch, "epochs": args.epochs,
                   "warm_start": {"ckpt": os.path.basename(E1_CKPT), "epoch": int(e1["epoch"]),
                                  "val_full": float(e1["val_full"])},
                   "firewall": fw, "history": hist, "best_epoch": best["epoch"] if best else None,
                   "minutes": round((time.time() - t00) / 60, 2)}, fh, indent=2, default=float)
    log(f"[done] {round((time.time() - t00) / 60, 2)} minutes; best epoch "
        f"{best['epoch'] if best else None}")


def mode_evaluate(args):
    """The gates (L.2), all on HOLDOUT users, all paired, all with the falsifier in view."""
    assert args.i_have_author_goahead, "REFUSING TO RUN without --i_have_author_goahead"
    from strategy_channel_suite import genv_eval_multi
    from policy_asif import rollout as asif_rollout, fold_seqs
    torch.manual_seed(SEED); np.random.seed(SEED)
    t00 = time.time()
    S, sblob = load_states()
    teachers = sblob["teachers"]
    E, rows, Dva, pool = open_env(f"forest/eval_{args.arm}")
    P = len(pool)
    phi_t, norm = build_phi(E, pool)
    fit, hold, fw = firewall(rows)
    hs = set(int(x) for x in hold.tolist())
    qidx = {q: j for j, q in enumerate(pool)}

    stem = f"forest_{args.arm}" + (f"_{args.tag}" if args.tag else "")
    blob = torch.load(os.path.join(FOREST_DIR, stem + ".pt"), map_location="cpu")
    assert [(int(a), int(b)) for a, b in blob["pool"]] == pool, "the ckpt is on another pool"
    assert np.array_equal(blob["norm_mu"], norm[0]) and np.array_equal(blob["norm_sd"], norm[1])
    pol = Policy(E["ctx"].d, phi_t.shape[1], P, arch="equivariant", state="both", use_pos=True)
    pol.load_state_dict(blob["model"])
    pol.eval()
    e1pol = Policy(E["ctx"].d, phi_t.shape[1], P, arch="equivariant", state="both", use_pos=True)
    e1b = torch.load(E1_CKPT, map_location="cpu")
    e1pol.load_state_dict(e1b["model"]); e1pol.eval()
    log(f"[policy] {stem}.pt epoch {blob['epoch']} arm {blob['arm']} -- FROZEN")

    z1 = np.load(E1_PER_USER)
    e1_full = np.asarray(z1["policy_full"], np.float64)
    e1_tail = np.asarray(z1["policy_tail"], np.float64)
    assert abs(float(e1_full.mean()) - E1_VAL_FULL) < 1e-6, "the banked E1 per-user vector moved"
    hpos = np.asarray([j for j, r in enumerate(np.asarray(Dva["uid"], np.int64)) if int(r) in hs],
                      np.int64)
    hrows = [int(r) for r in np.asarray(Dva["uid"], np.int64)[hpos]]

    out = {"analysis": "forest_evaluate", "design_sheet":
           "docs/design/PAPER_B_POLICY_DESIGN.md L.2", "citable": True, "test_touched": False,
           "split": "validation", "arm": args.arm, "ckpt_epoch": int(blob["epoch"]),
           "teachers": teachers, "firewall": fw, "n_holdout_users": len(hrows)}

    # ---- ★ THE FALSIFIER: the two fidelities, on HOLDOUT, side by side --------------------------
    hix = np.flatnonzero(np.asarray([int(r) in hs for r in S["row"]]))
    fT0 = fidelity_T0(pol, phi_t, Dva, P, hpos)
    fT1, fT1p = fidelity_T1(pol, phi_t, S, P, hix)
    eT0 = fidelity_T0(e1pol, phi_t, Dva, P, hpos)
    eT1, eT1p = fidelity_T1(e1pol, phi_t, S, P, hix)
    out["interference"] = {
        "T0_fidelity_forest": fT0, "T0_fidelity_E1": eT0, "T0_cost": fT0 - eT0,
        "T1_fidelity_forest": fT1, "T1_fidelity_E1": eT1, "T1_gain": fT1 - eT1,
        "T1_per_teacher_forest": {teachers[k]: v for k, v in fT1p.items()},
        "T1_per_teacher_E1": {teachers[k]: v for k, v in eT1p.items()},
        "verdict": ("INTERFERENCE: fidelity to T1 cost fidelity to T0" if fT0 < eT0 - 1e-12
                    else "no interference: T0 fidelity held while T1 fidelity moved")}
    log(f"[★ falsifier] T0 fidelity {fT0:.4f} (E1 {eT0:.4f}, cost {fT0 - eT0:+.4f}) | "
        f"T1 fidelity {fT1:.4f} (E1 {eT1:.4f}, gain {fT1 - eT1:+.4f}) -- {out['interference']['verdict']}")

    # ---- SPANNING: the cold-state rollout on HOLDOUT, against banked E1 on the same users --------
    ro = unseeded_rollout(E, pol, phi_t, pool, hrows)
    sf, st = ro["curves"][TOTAL_L]
    out["spanning"] = {
        "forest_full": float(np.nanmean(sf)), "forest_tail": float(np.nanmean(st)),
        "E1_full": float(e1_full[hpos].mean()), "E1_tail": float(e1_tail[hpos].mean()),
        "diff_full": paired_bootstrap(np.asarray(sf, np.float64) - e1_full[hpos]),
        "diff_tail": paired_bootstrap(np.asarray(st, np.float64) - e1_tail[hpos]),
        "frac_users_identical_to_E1": float(np.mean(np.abs(np.asarray(sf, np.float64)
                                                           - e1_full[hpos]) < 1e-12)),
        "n_distinct_interviews": len({tuple(r) for r in ro["asked"]}),
        "gate": "cold-state behaviour must not fall below E1 (HARD RULE 8)"}
    log(f"[spanning] forest {float(np.nanmean(sf)):.5f} vs E1 {float(e1_full[hpos].mean()):.5f} "
        f"diff {out['spanning']['diff_full']['mean']:+.6f} "
        f"[{out['spanning']['diff_full']['lo']:+.6f}, {out['spanning']['diff_full']['hi']:+.6f}]")

    # ---- BEHAVIOURAL: genre-seeded HOLDOUT states, policy vs its own teacher ---------------------
    from group_greedy_genre import load_item_genres, partition_by_genre
    orders, _bs, _gb = load_group_orders()
    gen, _v = load_item_genres(E["ctx"].ni)
    groups, _a, _b2 = partition_by_genre(E, rows, gen)
    gmap = {g: int(sblob["per_teacher"][g]["concept_id"]) for g in teachers}
    beh = {}
    for g in teachers:
        grows = [int(r) for r in sorted(groups[g]) if int(r) in hs]
        if len(grows) < args.min_eval_group:
            beh[g] = {"n_holdout_users": len(grows), "skipped": "below the reporting floor"}
            log(f"[behaviour {g}] {len(grows)} holdout users -- below the reporting floor, skipped")
            continue
        seeds = [[(1, gmap[g])] for _ in grows]
        seed_idx = [[qidx[(1, gmap[g])]] for _ in grows]
        from ndcg_tree import branch_of_per_user
        oo = branch_of_per_user(E, grows, [(1, gmap[g])] * len(grows))
        seed_out = [[int(x)] for x in oo]
        r_f = asif_rollout(E, pol, phi_t, phi_t, pool, grows, seeds, seed_idx, seed_out, "asif")
        f_f, t_f = r_f["curves"][TOTAL_L]
        r_e = asif_rollout(E, e1pol, phi_t, phi_t, pool, grows, seeds, seed_idx, seed_out, "asif")
        f_e, t_e = r_e["curves"][TOTAL_L]
        t1_seq = [[(0, int(i)) for i in orders[g]] for _ in grows]
        f_t1, t_t1 = genv_eval_multi(E, grows, *fold_seqs(seeds, t1_seq))
        tr_asked = np.asarray(Dva["label_idx"], np.int64)
        pos_of = {int(r): j for j, r in enumerate(np.asarray(Dva["uid"], np.int64))}
        tr_seq = [[pool[int(j)] for j in tr_asked[pos_of[r], :TOTAL_L - 1]] for r in grows]
        f_tr, t_tr = genv_eval_multi(E, grows, *fold_seqs(seeds, tr_seq))
        follows = float(np.mean([np.array_equal(
            np.asarray(r_f["asked"][u], np.int64),
            np.asarray([qidx[(0, int(i))] for i in orders[g]], np.int64))
            for u in range(len(grows))]))
        beh[g] = {
            "n_holdout_users": len(grows),
            "forest_full": float(np.nanmean(f_f)), "forest_tail": float(np.nanmean(t_f)),
            "E1_full": float(np.nanmean(f_e)), "E1_tail": float(np.nanmean(t_e)),
            "T1_order_full": float(np.nanmean(f_t1)), "T1_order_tail": float(np.nanmean(t_t1)),
            "tree_budget_full": float(np.nanmean(f_tr)), "tree_budget_tail": float(np.nanmean(t_tr)),
            "forest_minus_T1order": {"full": paired_bootstrap(np.asarray(f_f, np.float64)
                                                              - np.asarray(f_t1, np.float64)),
                                     "tail": paired_bootstrap(np.asarray(t_f, np.float64)
                                                              - np.asarray(t_t1, np.float64))},
            "forest_minus_tree": {"full": paired_bootstrap(np.asarray(f_f, np.float64)
                                                           - np.asarray(f_tr, np.float64)),
                                  "tail": paired_bootstrap(np.asarray(t_f, np.float64)
                                                           - np.asarray(t_tr, np.float64))},
            "forest_minus_E1": {"full": paired_bootstrap(np.asarray(f_f, np.float64)
                                                         - np.asarray(f_e, np.float64)),
                                "tail": paired_bootstrap(np.asarray(t_f, np.float64)
                                                         - np.asarray(t_e, np.float64))},
            "forest_reproduces_the_group_order_exactly": follows,
            "n_distinct_forest_interviews": len({tuple(r) for r in r_f["asked"]})}
        log(f"[behaviour {g}] n={len(grows)} forest {beh[g]['forest_full']:.5f} | T1 order "
            f"{beh[g]['T1_order_full']:.5f} | tree {beh[g]['tree_budget_full']:.5f} | E1 "
            f"{beh[g]['E1_full']:.5f} | reproduces its order exactly on {follows * 100:.2f}%")
    out["behavioural"] = beh
    os.makedirs(FOREST_DIR, exist_ok=True)
    p = os.path.join(FOREST_DIR, stem + "_gates.json")
    out["minutes"] = round((time.time() - t00) / 60, 2)
    with open(p, "w") as fh:
        json.dump(out, fh, indent=2, default=float)
    log(f"[done] wrote {p} ({out['minutes']:.1f}m)")


def main():
    ap = argparse.ArgumentParser(description="Experiment L2a (Part L.2): the two-teacher forest")
    ap.add_argument("--mode", required=True, choices=("composition", "states", "fit", "evaluate"))
    ap.add_argument("--arm", default="main", choices=("main", "shuffle"))
    ap.add_argument("--tag", default="")
    ap.add_argument("--min_group", type=int, default=MIN_GROUP)
    ap.add_argument("--min_eval_group", type=int, default=50)
    ap.add_argument("--t1_frac", type=float, default=0.5)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--full_threads", action="store_true")
    ap.add_argument("--i_have_author_goahead", action="store_true")
    args = ap.parse_args()
    assert "--split" not in sys.argv, "there is no --split here: this file is validation-only"
    log("#" * 110)
    log(f"# EXPERIMENT L2a -- THE FOREST, mode={args.mode} arm={args.arm}. VALIDATION ONLY.")
    log("# The 10,000 canonical TEST users are NOT built, NOT loaded, NOT touched.")
    log("#" * 110)
    {"composition": mode_composition, "states": mode_states, "fit": mode_fit,
     "evaluate": mode_evaluate}[args.mode](args)


if __name__ == "__main__":
    main()
