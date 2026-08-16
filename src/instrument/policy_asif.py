r"""EXPERIMENT L1 -- THE AS-IF-ASKED ENCODING TEST.

Design sheet: `docs/design/PAPER_B_POLICY_DESIGN.md` **PART L.1 is binding**. Committed before the
run (HARD RULE 10). VALIDATION ONLY -- `--split` is not an option here, the env is always
`validation`, and the 10,000 canonical TEST users are never built, never loaded, never named.

WHAT IS BEING ASKED
-------------------
Experiment R fed volunteered evidence into the tower latent `z` ONLY, leaving the explicit ternary
asked-history EMPTY. The policy therefore saw a state that says "nothing has been asked" and ran its
unseeded interview for 95-99% of users. L.1's claim: a volunteered STATED fact is *an answered
question nobody asked*, and it must enter through the SAME channel an asked question does --

  (a) as a history token `phi_q (x) ternary outcome`,   (b) folded into `z`,   (c) MASKED from
      re-asking (the answer is already held; re-asking it burns a turn for nothing).

R did (b) alone. This file does all three, and measures what changes.

THE THREE CELLS (L.1)
---------------------
  1. root-matching   the volunteered fact IS the tree's root splitter, concept (1, 114). This is the
                     stated-genre case. ** FALSIFIABLE PREDICTION: the policy branches exactly as the
                     tree's post-root arm -- E1's 1.0000 fidelity guarantees it IF the encoding is
                     right. If cell 1 fails, the ENCODING is wrong, not the theory. **
  2. pool-nontree    volunteered facts are pool CONCEPTS that are not tree splitters. Genuinely new
                     behaviour: no teacher ever put the policy in this state.
  3. items           R's own popularity-weighted item draw (k in {2,4,8}, seed 98765), re-encoded as
                     history tokens. Nearest-manifold behaviour, and the direct A/B against R.

THE BUDGET, DECLARED (HARD RULE 1)
----------------------------------
Every arm in a cell holds the SAME number of evidence questions: **16 in total**, of which `s_u` are
the volunteered ones. The policy asks `16 - s_u`. This is the only budget under which the arms are
comparable, and it is also the only one that keeps the position feature `(t, 16 - t)` inside the
range E1 was trained on. `tree_full16` is reported ALONGSIDE as the *strong* comparator -- the tree
given its full 16 questions on top of the volunteered evidence (R's protocol, 16 + s questions of
evidence). It is a handicap in the tree's favour and is labelled as such wherever it appears.

THE ARMS
--------
  none          the volunteered evidence alone; asks nothing. Isolates what the interview adds.
  tree_budget   the volunteered evidence + the tree's first `16 - s_u` questions from its root. The
                tree cannot route on volunteered facts and cannot skip a question it already knows
                the answer to -- that is the mechanism, made visible as a redundancy count.
  tree_full16   the volunteered evidence + all 16 of the tree's questions. HANDICAP IN ITS FAVOUR.
  zonly_enc     R's encoding at the matched budget: evidence folded into `z`, history EMPTY, nothing
                masked. The A/B partner -- it differs from `asif` in the ENCODING and nothing else.
  asif          the L.1 encoding: history token + fold + mask.

REDUNDANCY IS AN ASSERTION, NOT A MEASUREMENT, FOR `asif`
---------------------------------------------------------
`asif` masks every volunteered pool question out of the candidate table, so it CANNOT re-ask one.
The count is asserted to be exactly 0, which is what validates the mask wiring. For the other arms
it is a real measurement.
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
                        E1_VAL_FULL, E1_PER_USER, Policy, build_phi, paired_bootstrap,
                        channel_mix, load_traj, log)

ASIF_DIR = os.path.join(OUTDIR, "asif")
ROOT_Q = (1, 114)                       # the tree's root splitter, read from ndcgtree16.json below
ARMS = ("none", "tree_budget", "tree_full16", "zonly_enc", "asif")


# ================================================================================ small helpers
def dedup_keep_order(seq):
    seen, out = [], []
    for x in seq:
        if x not in seen:
            seen.append(x); out.append(x)
    return out


def split_by_channel(qs):
    return ([int(i) for ch, i in qs if int(ch) == 0], [int(i) for ch, i in qs if int(ch) == 1])


def fold_seqs(seeds, asked_qs):
    """(item_seqs, concept_seqs) for `seeds` (volunteered, they arrive first) then `asked_qs`.
    Duplicates are folded ONCE -- a re-ask burns the turn and delivers no new evidence."""
    iseqs, cseqs = [], []
    for sd, aq in zip(seeds, asked_qs):
        it, cc = split_by_channel(list(sd) + list(aq))
        iseqs.append(dedup_keep_order(it)); cseqs.append(dedup_keep_order(cc))
    return iseqs, cseqs


def redundancy(seeds, asked_qs):
    return np.asarray([sum(1 for q in aq if tuple(q) in {tuple(x) for x in sd})
                       for sd, aq in zip(seeds, asked_qs)], np.int64)


def first_divergence(asked, cont, n_ask):
    """Position of the first question at which the policy leaves the tree's continuation, per user.
    -1 means it never does (over that user's whole realised budget)."""
    n = len(asked)
    out = np.full(n, -1, np.int64)
    for u in range(n):
        for t in range(int(n_ask[u])):
            if int(asked[u][t]) != int(cont[u][t]):
                out[u] = t
                break
    return out


# ================================================================================ the seed draws
def seeds_root(E, rows, pool):
    """CELL 1. Every user volunteers the tree's root concept."""
    return [[ROOT_Q] for _ in rows]


def seeds_pool_nontree(E, rows, pool, splitters, s=2, seed=SEED):
    """CELL 2. `s` pool CONCEPTS per user, drawn uniformly WITHOUT REPLACEMENT from the concepts the
    user can actually answer (`E["foldable"]`) and that are NOT tree splitters. Restricting to
    answerable concepts is the same choice R made for items (it drew from the user's rated fold-in
    set): a volunteered fact is by construction one the person can state. Targets are excluded from
    `foldable` by construction, so nothing here can see a target."""
    cand = np.asarray([int(i) for ch, i in pool if int(ch) == 1 and (1, int(i)) not in splitters],
                      np.int64)
    rng = np.random.default_rng(seed)
    F = E["foldable"]
    out = []
    for r in rows:
        ok = cand[np.asarray(F[int(r), cand], bool)]
        take = int(min(s, len(ok)))
        sel = rng.choice(len(ok), size=take, replace=False) if take else []
        out.append([(1, int(ok[j])) for j in sel])
    return out


def seeds_items(E, rows, k, seed=SEED):
    """CELL 3. R's own draw, byte-for-byte: `policy_resume.draw_volunteered`."""
    from policy_resume import draw_volunteered
    vol, info = draw_volunteered(E, rows, k, seed=seed)
    return [[(0, int(i)) for i in v] for v in vol], info


# ================================================================================ the rollout
def rollout(E, pol, phi_all, phi_cand, pool, rows, seeds, seed_idx, seed_out, encoding,
            total=TOTAL_L, batch=1000):
    """One seeded interview per user, under `encoding` in {"asif", "zonly_enc"}.

    asif       the seeds occupy history slots 0..s_u-1 (phi (x) their true ternary outcome), their
               pool entries are masked out of the candidate table, and the position feature counts
               them: the state says "s_u questions asked, 16 - s_u to go", which is exactly the
               state E1 was trained on.
    zonly_enc  R's encoding: the seeds move `z` and nothing else. History starts empty and the
               position feature counts only the policy's own asks.

    In both arms the policy asks `16 - s_u` questions, so both hold 16 evidence questions at the end.
    """
    assert encoding in ("asif", "zonly_enc")
    from strategy_channel_suite import genv_eval_multi
    from ndcg_tree import branch_of_per_user
    n, P, T = len(rows), len(pool), TOTAL_L - 1
    ns = np.asarray([len(s) for s in seeds], np.int64)
    n_ask = total - ns
    assert (n_ask >= 1).all(), "a user has no budget left to ask anything"
    max_turns = int(n_ask.max())

    asked = np.full((n, max_turns), -1, np.int64)
    outs = np.full((n, max_turns), -1, np.int8)
    hq = np.zeros((n, T), np.int64); ho = np.zeros((n, T), np.int64); hm = np.zeros((n, T), np.float32)
    asked_mask = np.zeros((n, P), bool)
    if encoding == "asif":
        for u in range(n):
            for j, (pi, oo) in enumerate(zip(seed_idx[u], seed_out[u])):
                hq[u, j] = int(pi); ho[u, j] = int(oo); hm[u, j] = 1.0
                if int(pi) < P:                     # an out-of-pool seed cannot be re-asked anyway
                    asked_mask[u, int(pi)] = True

    asked_qs = [[] for _ in range(n)]
    iseq, cseq = fold_seqs(seeds, asked_qs)
    z = genv_eval_multi(E, rows, iseq, cseq, score=False, return_z=True)[-1].numpy()
    phi_g = torch.from_numpy(phi_cand)
    pol.eval()
    curves = {}
    for t in range(max_turns):
        act = np.where(n_ask > t)[0]
        base = ns if encoding == "asif" else np.zeros_like(ns)
        pt = (base + t).astype(np.float32)
        pos = np.stack([pt, TOTAL_L - pt], 1)
        with torch.no_grad():
            for st in range(0, n, batch):
                sl = slice(st, min(st + batch, n))
                s = pol(torch.from_numpy(z[sl]), torch.from_numpy(phi_all[hq[sl]]),
                        torch.from_numpy(ho[sl]), torch.from_numpy(hm[sl]),
                        torch.from_numpy(pos[sl]), phi_g, torch.from_numpy(asked_mask[sl]))
                asked[sl, t] = s.argmax(1).numpy()
        asked[n_ask <= t, t] = -1
        qs = [pool[int(asked[u, t])] if asked[u, t] >= 0 else pool[0] for u in range(n)]
        oo = branch_of_per_user(E, rows, qs)
        for u in act:
            j = int(asked[u, t])
            outs[u, t] = oo[u]
            asked_qs[u].append(pool[j])
            asked_mask[u, j] = True
            slot = int(base[u] + t)
            hq[u, slot] = j; ho[u, slot] = int(oo[u]); hm[u, slot] = 1.0
        iseq, cseq = fold_seqs(seeds, asked_qs)
        need = (t + 1 == max_turns)
        r = genv_eval_multi(E, rows, iseq, cseq, return_z=(not need), score=need)
        if need:
            curves[total] = (r[0], r[1])
        else:
            z = r[-1].numpy()
        log(f"    [rollout {encoding}] turn {t + 1}/{max_turns} done")
    for u in range(n):
        assert len(set(map(tuple, asked_qs[u]))) == len(asked_qs[u]), \
            "a question was asked twice inside one interview"
    return {"asked": asked, "outs": outs, "asked_qs": asked_qs, "curves": curves, "n_ask": n_ask}


# ================================================================================ persistence
def cell_path(cell, arm):
    return os.path.join(ASIF_DIR, f"asif_{cell}_{arm}.json")


def write_cell(cell, arm, blob, per_user):
    os.makedirs(ASIF_DIR, exist_ok=True)
    p = cell_path(cell, arm)
    np.savez_compressed(p.replace(".json", "_per_user.npz"), **per_user)
    with open(p + ".tmp", "w") as fh:
        json.dump(blob, fh, indent=2, default=float)
    os.replace(p + ".tmp", p)
    log(f"  [cell] wrote {os.path.basename(p)}")


def read_cell(cell, arm, sig):
    p = cell_path(cell, arm)
    if not os.path.exists(p):
        return None
    b = json.load(open(p))
    if b.get("contract") != sig:
        log(f"  [cell] {os.path.basename(p)} is under a DIFFERENT contract -- refusing to resume")
        return None
    z = np.load(p.replace(".json", "_per_user.npz"), allow_pickle=True)
    return b, {k: z[k] for k in z.files}


# ================================================================================ the run
def main():
    ap = argparse.ArgumentParser(description="Experiment L1 (Part L.1): the as-if-asked encoding")
    ap.add_argument("--cells", default="root,nontree,items2,items4,items8")
    ap.add_argument("--ckpt", default=os.path.join(OUTDIR, "policy_main_equivariant_both.pt"))
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--full_threads", action="store_true")
    ap.add_argument("--i_have_author_goahead", action="store_true")
    args = ap.parse_args()
    assert args.i_have_author_goahead, "REFUSING TO RUN without --i_have_author_goahead"
    cells = [c.strip() for c in args.cells.split(",") if c.strip()]
    t00 = time.time()
    os.makedirs(ASIF_DIR, exist_ok=True)

    from strategy_channel_suite import (build_greedy_env, contract_selftest, genv_eval, genv_mean,
                                        genv_eval_multi, CERT_TOWER, CERT_CONC)
    from ndcg_tree import build_pool, leak_assert, branch_of_selftest, branch_of_per_user
    torch.manual_seed(SEED); np.random.seed(SEED)

    log("#" * 110)
    log("# EXPERIMENT L1 -- THE AS-IF-ASKED ENCODING (design sheet Part L.1). VALIDATION ONLY.")
    log("# The 10,000 canonical TEST users are NOT built, NOT loaded, NOT touched.")
    log("#" * 110)

    E = build_greedy_env(CERT_TOWER, "validation", CERT_CONC, "asif")
    contract_selftest(E, E["rows"])
    leak_assert(E, E["rows"], "asif/val")
    rows = E["rows"]
    assert len(rows) == 10000, f"validation cohort is {len(rows)} users, expected 10,000"

    f0, t0_ = genv_mean(E, [], rows)
    assert round(f0, 8) == VAL_M0_FULL, f"CANONICAL SNAP FAILED: {f0!r} != {VAL_M0_FULL}"
    log(f"[gate] canonical m=0 snap full={f0!r} tail={t0_!r} -- exact")

    Dva = load_traj(train=False)
    pool = Dva["pool"]
    pool_e, _ip, _cp = build_pool(E, 500, 500)
    assert pool_e == pool, "the env's pool is not the trajectory bank's pool"
    assert list(np.asarray(Dva["uid"], np.int64)) == list(np.asarray(rows, np.int64)), \
        "the validation trajectory rows are not the env's rows, in order"
    branch_of_selftest(E, rows[:2000], [pool[0], pool[500], ROOT_Q])

    static_seq = [(int(a), int(b)) for a, b in json.load(open(STATIC_JSON))["sequence"]]
    sf, _stl = genv_eval(E, static_seq, rows)
    assert abs(float(np.nanmean(sf)) - STATIC_VAL_FULL) < 1e-12, "BANKED-HEADLINE ASSERT FAILED"
    log(f"[gate] greedy_static_policyB_combined reproduces EXACTLY: full={float(np.nanmean(sf))!r}")

    tree = json.load(open(PN.TREE16_JSON))
    splitters = {(int(a), int(b)) for L in tree["leaf_table"] for a, b in L["path"]}
    assert ROOT_Q in splitters and len({tuple(L["path"][0]) for L in tree["leaf_table"]}) == 1, \
        "the tree's root is not unique"
    assert tuple(tree["leaf_table"][0]["path"][0]) == ROOT_Q, "the tree's root moved"
    log(f"[tree] root {ROOT_Q}, {len(splitters)} splitter questions: {sorted(splitters)}")

    z1 = np.load(E1_PER_USER)
    e1_full = np.asarray(z1["policy_full"], np.float64)
    e1_tail = np.asarray(z1["policy_tail"], np.float64)
    e1_asked = np.asarray(z1["asked"], np.int64)
    assert abs(float(e1_full.mean()) - E1_VAL_FULL) < 1e-6, "the banked E1 per-user vector moved"
    tree_asked = np.asarray(Dva["label_idx"], np.int64)
    assert np.array_equal(e1_asked, tree_asked), "E1's banked rollout is not the tree's interview"
    root_idx = pool.index(ROOT_Q)
    assert (tree_asked[:, 0] == root_idx).all(), "the tree does not open with its root for every user"
    log(f"[gate] E1 == the tree: banked per-user mean full {float(e1_full.mean()):.9f}; "
        f"root pool index {root_idx}")

    blob = torch.load(args.ckpt, map_location="cpu")
    assert [(int(a), int(b)) for a, b in blob["pool"]] == pool, "the ckpt was fitted on another pool"
    phi_t, norm = build_phi(E, pool)
    assert np.array_equal(blob["norm_mu"], norm[0]) and np.array_equal(blob["norm_sd"], norm[1]), \
        "the ckpt's descriptor standardisation differs from this env's"
    pol = Policy(E["ctx"].d, phi_t.shape[1], len(pool), arch=blob["arch"], state=blob["state"],
                 use_pos=blob["use_pos"])
    pol.load_state_dict(blob["model"])
    assert blob["state"] == "both" and blob["use_pos"], "L1 needs the both-inputs, positional E1"
    log(f"[policy] {os.path.basename(args.ckpt)} epoch {blob['epoch']} val_full {blob['val_full']:.6f}"
        f" -- FROZEN, no gradient step is taken anywhere in this file")

    # R's banked cells, snapshotted for the free chain check (cell 3's tree_full16 IS R's tree arm)
    banked_R = {}
    _rp = os.path.join(OUTDIR, "resume", "policy_resume.json")
    if os.path.exists(_rp):
        _b = json.load(open(_rp))
        for _k, _v in _b.get("by_k", {}).items():
            for _a, _c in _v.get("arms", {}).items():
                banked_R[(_k, _a)] = (_c["full"], _c["tail"])
        log(f"[banked-R] snapshotted {len(banked_R)} (k, arm) cells from R for assertion")

    sig = {"tower": E["tower"], "conc_ckpt": E["conc_ckpt"], "pool_contract": E["pool_contract"],
           "ckpt": os.path.basename(args.ckpt), "seed": SEED, "L": TOTAL_L, "budget": "matched-16"}

    out_all = {}
    for cell in cells:
        log("=" * 100)
        log(f"=== CELL {cell}")
        draw_info = {}
        if cell == "root":
            seeds = seeds_root(E, rows, pool); tree_off = 1
        elif cell == "nontree":
            seeds = seeds_pool_nontree(E, rows, pool, splitters, s=2); tree_off = 0
        elif cell.startswith("items"):
            seeds, draw_info = seeds_items(E, rows, int(cell[5:])); tree_off = 0
        else:
            raise SystemExit(f"unknown cell {cell!r}")
        ns = np.asarray([len(s) for s in seeds], np.int64)
        assert (ns >= 1).all(), "a user has no volunteered evidence at all"

        # ---- the seeds' pool indices, their descriptors and their true ternary outcomes ----------
        extra = sorted({tuple(q) for sd in seeds for q in sd} - set(pool))
        if extra:
            phi_x, _ = build_phi(E, extra, norm=norm)
            phi_all = np.concatenate([phi_t, phi_x], 0)
        else:
            phi_all = phi_t
        qidx = {q: j for j, q in enumerate(pool)}
        qidx.update({q: len(pool) + j for j, q in enumerate(extra)})
        seed_idx = [[qidx[tuple(q)] for q in sd] for sd in seeds]
        smax = int(ns.max())
        seed_out = [[] for _ in rows]
        for j in range(smax):
            qs = [tuple(seeds[u][j]) if j < ns[u] else pool[0] for u in range(len(rows))]
            oo = branch_of_per_user(E, rows, qs)
            for u in range(len(rows)):
                if j < ns[u]:
                    seed_out[u].append(int(oo[u]))
        flat_out = np.asarray([o for so in seed_out for o in so], np.int64)
        comp = {"n_users": len(rows), "seed_size_mean": float(ns.mean()),
                "seed_size_histogram": {str(int(v)): int(c) for v, c in
                                        zip(*np.unique(ns, return_counts=True))},
                "n_distinct_seed_questions": len({tuple(q) for sd in seeds for q in sd}),
                "n_seed_questions_out_of_pool": len(extra),
                "seed_outcome_mix": {k: float((flat_out == v).mean())
                                     for k, v in PN.OUTCOME.items()},
                "policy_ask_budget_mean": float((TOTAL_L - ns).mean()), "draw": draw_info}
        log(f"[composition] {json.dumps(comp, default=float)}")

        cells_out = {}
        for arm in ARMS:
            got = read_cell(cell, arm, sig)
            if got is not None:
                log(f"  [resume] {cell} {arm} already banked -- reusing")
                cells_out[arm] = got
                continue
            t0 = time.time()
            n_ask = TOTAL_L - ns
            if arm == "none":
                aq = [[] for _ in rows]
                f, t = genv_eval_multi(E, rows, *fold_seqs(seeds, aq))
                asked = np.full((len(rows), 0), -1, np.int64); nq = np.zeros(len(rows), np.int64)
            elif arm in ("tree_budget", "tree_full16"):
                nq = n_ask if arm == "tree_budget" else np.full(len(rows), TOTAL_L, np.int64)
                aq = [[pool[int(j)] for j in tree_asked[u, :int(nq[u])]] for u in range(len(rows))]
                f, t = genv_eval_multi(E, rows, *fold_seqs(seeds, aq))
                asked = tree_asked[:, :int(nq.max())]
            else:
                ro = rollout(E, pol, phi_all, phi_t, pool, rows, seeds, seed_idx, seed_out,
                             "asif" if arm == "asif" else "zonly_enc")
                aq = ro["asked_qs"]; asked = ro["asked"]; nq = ro["n_ask"]
                f, t = ro["curves"][TOTAL_L]
            red = redundancy(seeds, aq)
            if arm == "asif":
                assert int(red.sum()) == 0, (
                    f"MASK WIRING BROKEN: {int(red.sum())} re-asks of a volunteered question under "
                    f"the as-if-asked encoding, which masks them out of the candidate table")
            blob_c = {
                "analysis": "policy_asif_cell", "design_sheet":
                "docs/design/PAPER_B_POLICY_DESIGN.md L.1", "test_touched": False,
                "split": "validation", "cell": cell, "arm": arm, "contract": sig,
                "composition": comp,
                "full": float(np.nanmean(f)), "tail": float(np.nanmean(t)),
                "n_questions_asked_mean": float(nq.mean()),
                "n_evidence_questions_mean": float(np.mean([len(set(map(tuple, list(sd) + list(a))))
                                                            for sd, a in zip(seeds, aq)])),
                "redundant_reasks_mean": float(red.mean()),
                "redundant_reasks_total": int(red.sum()),
                "frac_users_with_a_redundant_reask": float((red > 0).mean()),
                "concept_share": (float(np.mean([np.mean([int(q[0]) == 1 for q in a]) for a in aq
                                                 if len(a)])) if any(len(a) for a in aq) else None),
                "n_distinct_interviews": len({tuple(map(tuple, a)) for a in aq}),
                "minutes": round((time.time() - t0) / 60, 2)}
            pu = {"full": f, "tail": t, "asked": asked, "redundant": red, "n_ask": nq,
                  "seed_sizes": ns, "rows": np.asarray(rows, np.int64)}
            write_cell(cell, arm, blob_c, pu)
            cells_out[arm] = (blob_c, pu)
            log(f"  [{cell} {arm}] full={blob_c['full']:.5f} tail={blob_c['tail']:.5f} "
                f"redundant={blob_c['redundant_reasks_mean']:.4f} "
                f"distinct={blob_c['n_distinct_interviews']} ({blob_c['minutes']:.1f}m)")

        gf = {a: np.asarray(cells_out[a][1]["full"], np.float64) for a in ARMS}
        gt = {a: np.asarray(cells_out[a][1]["tail"], np.float64) for a in ARMS}

        # ---- the free chain checks -------------------------------------------------------------
        chain = {}
        if cell == "root":
            d = np.abs(gf["tree_full16"] - e1_full)
            chain["tree_full16_is_banked_E1"] = {"max_abs_per_user_diff": float(d.max()),
                                                 "mean_this_run": float(gf["tree_full16"].mean()),
                                                 "banked": float(e1_full.mean())}
            assert d.max() < 1e-12, (
                "the root cell's tree_full16 is the tree's own 16 questions with its root folded "
                "twice -- it MUST reproduce banked E1 per user, and it does not")
            log("[chain] root cell tree_full16 == banked E1 per user, exactly")
        if cell.startswith("items"):
            k = cell[5:]
            for arm, want_key in (("tree_full16", "tree"), ("none", "none")):
                want = banked_R.get((k, want_key))
                if want is None:
                    continue
                got = (cells_out[arm][0]["full"], cells_out[arm][0]["tail"])
                assert abs(got[0] - want[0]) < 1e-12 and abs(got[1] - want[1]) < 1e-12, (
                    f"BANKED-R ASSERT FAILED at k={k} {want_key}: this run {got!r}, R published {want!r}")
                chain[f"R_{want_key}_k{k}_reproduces"] = {"this_run": got, "R": want}
                log(f"[chain] R's {want_key} arm at k={k} reproduces exactly: {got!r}")

        # ---- the falsifier and the behaviour diagnostics ----------------------------------------
        n_ask = TOTAL_L - ns
        a_asif = cells_out["asif"][1]["asked"]
        a_zonly = cells_out["zonly_enc"][1]["asked"]
        cont = [tree_asked[u, tree_off:tree_off + int(n_ask[u])] for u in range(len(rows))]
        follows = np.asarray([bool(np.array_equal(np.asarray(a_asif[u][:int(n_ask[u])], np.int64),
                                                  np.asarray(cont[u], np.int64)))
                              for u in range(len(rows))])
        follows_z = np.asarray([bool(np.array_equal(np.asarray(a_zonly[u][:int(n_ask[u])], np.int64),
                                                    np.asarray(cont[u], np.int64)))
                                for u in range(len(rows))])
        fd = first_divergence(a_asif, cont, n_ask)
        fdz = first_divergence(a_zonly, cont, n_ask)
        beh = {
            "tree_continuation": ("the tree's questions from position "
                                  f"{tree_off} (its root is the volunteered fact)" if tree_off
                                  else "the tree's own questions from its root"),
            "asif_follows_tree_continuation_exactly": float(follows.mean()),
            "zonly_follows_tree_continuation_exactly": float(follows_z.mean()),
            "asif_first_divergence_histogram": {str(int(v)): int(c) for v, c in
                                                zip(*np.unique(fd, return_counts=True))},
            "zonly_first_divergence_histogram": {str(int(v)): int(c) for v, c in
                                                 zip(*np.unique(fdz, return_counts=True))},
            "asif_mean_first_divergence_among_divergers":
                float(fd[fd >= 0].mean()) if (fd >= 0).any() else None,
            "n_distinct_asif_interviews": cells_out["asif"][0]["n_distinct_interviews"],
            "n_distinct_zonly_interviews": cells_out["zonly_enc"][0]["n_distinct_interviews"]}

        cmp_pairs = [("asif", "tree_budget"), ("asif", "tree_full16"), ("asif", "zonly_enc"),
                     ("asif", "none"), ("zonly_enc", "tree_budget"), ("tree_budget", "none"),
                     ("tree_full16", "tree_budget")]
        comps = {f"{a}_minus_{b}": {"full": paired_bootstrap(gf[a] - gf[b]),
                                    "tail": paired_bootstrap(gt[a] - gt[b])}
                 for a, b in cmp_pairs}
        out_all[cell] = {"composition": comp, "tree_offset": tree_off,
                         "arms": {a: cells_out[a][0] for a in ARMS},
                         "chain_checks": chain, "behaviour": beh, "comparisons": comps}
        p = os.path.join(ASIF_DIR, "policy_asif.json")
        blob_o = {"analysis": "policy_asif_L1", "design_sheet":
                  "docs/design/PAPER_B_POLICY_DESIGN.md L.1", "citable": True, "test_touched": False,
                  "split": "validation", "n_users": len(rows), "contract": sig,
                  "budget": "every arm holds 16 evidence questions in total; tree_full16 is the "
                            "handicapped comparator with 16 asked ON TOP of the volunteered set",
                  "reference": {"E1_tree_full": float(e1_full.mean()),
                                "E1_tree_tail": float(e1_tail.mean()),
                                "m0_full": f0, "m0_tail": t0_},
                  "by_cell": out_all, "minutes": round((time.time() - t00) / 60, 2)}
        with open(p + ".tmp", "w") as fh:
            json.dump(blob_o, fh, indent=2, default=float)
        os.replace(p + ".tmp", p)
        c = comps["asif_minus_tree_budget"]["full"]
        log(f"[{cell}] asif follows the tree's continuation on {follows.mean() * 100:.2f}% of users "
            f"(z-only encoding: {follows_z.mean() * 100:.2f}%) | asif - tree_budget full "
            f"{c['mean']:+.5f} [{c['lo']:+.5f}, {c['hi']:+.5f}]")
    log(f"[done] {round((time.time() - t00) / 60, 2)} minutes")


if __name__ == "__main__":
    main()
