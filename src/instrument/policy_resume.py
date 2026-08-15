r"""EXPERIMENT R -- RESUMPTION: "the architecture supports new states".

Design sheet: `docs/design/PAPER_B_POLICY_DESIGN.md` **PART H.2 is binding**. Committed before the
run (HARD RULE 10). Validation only; the 10,000 canonical TEST users are never built, never loaded,
never named -- `--split` is not even an option here, the env is always `validation`.

WHAT IS BEING ASKED
-------------------
Every arm starts from a state it did not create: the user has ALREADY volunteered `k` items, which
are folded as ordinary evidence before the first question. Then each arm conducts a 16-question
interview from that seeded state.

  tree     cannot route on it. Its routing consumes the ternary answers to ITS OWN questions only,
           so the volunteered items cannot move which question it asks next. Its realised interview
           is bit-identical to the unseeded one -- asserted here, not assumed.
  static   cannot route on anything. Same 16 questions always.
  policy   reads the seeded latent `z_0`. Whether it adapts is THE RESULT, either way.
  none     asks nothing more. The k items alone. Isolates what the interview adds on top.

THE HONEST FORK, DECLARED BEFORE THE RUN (Part H.2): E1 never deviates from the tree on-manifold
(`frac_never_deviates = 1.0000`), so it may not deviate here either. If it adapts and beats the
tree, distillation generalises off the teacher's manifold. If it does not, that measures what
one-hot distillation fails to buy. FALSIFIER: if the policy adapts and STILL loses to the tree on
seeded states -- the setting built to disadvantage the tree -- the "supports new states" claim is
dead and is reported dead.

THE DRAW, STATED EXPLICITLY (HARD RULE 1: any sampling design is declared in the artifact)
------------------------------------------------------------------------------------------
For each validation user, `k` items are drawn WITHOUT REPLACEMENT from the items that user rated on
their FOLD-IN side -- exactly the set the item-answer rule can answer from (`E["lvl"][r]`, built
from `ctx.allb` with the held-out targets removed, so nothing here can touch a target). The draw is
POPULARITY-WEIGHTED (p proportional to the train count `ctx.cnt`), because a person volunteering
films names films people have seen; a uniform draw over a user's ratings would over-represent the
tail and make the seeded state unrealistically informative. One `numpy.default_rng(98765)` stream,
users consumed in canonical row order. A user with fewer than k rated fold-in items contributes all
of them; the histogram of realised seed sizes is reported.

REDUNDANCY, THE MECHANISM MADE VISIBLE
--------------------------------------
An asked ITEM question whose id is already in the volunteered set is a question the evidence has
already answered. It is counted per arm, and it is also DEDUPLICATED out of the folded sequence --
which is the honest semantics: the turn is burned and no new evidence arrives. That is precisely
the cost the tree cannot avoid and the policy is being asked whether it can.
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
from policy_net import (OUTDIR, SEED, TOTAL_L, STATIC_JSON, STATIC_VAL_FULL, STATIC_VAL_TAIL,
                        VAL_M0_FULL, E1_VAL_FULL, E1_PER_USER, STATIC_ITEMS_VAL_FULL,
                        STATIC_ITEMS_VAL_TAIL, Policy, build_phi, paired_bootstrap,
                        pep_divergence_from_arrays, channel_mix, load_traj, log)

RESUME_DIR = os.path.join(OUTDIR, "resume")
KS = (2, 4, 8)
ARMS = ("none", "static", "tree", "policy")


# =============================================================================== the volunteered draw
def draw_volunteered(E, rows, k, seed=SEED):
    """`k` items per user, popularity-weighted, without replacement, from the user's rated FOLD-IN
    items. Declared, not silent (module docstring). Returns (list of id-lists, diagnostics)."""
    lvl = E["lvl"]
    cnt = np.asarray(E["ctx"].cnt, np.float64)
    rng = np.random.default_rng(seed)
    out, sizes, n_liked, n_tot = [], [], 0, 0
    from ndcg_tree import LIKE_MIN_LEVEL
    for r in rows:
        d = lvl[int(r)]
        ids = np.fromiter(d.keys(), np.int64, len(d))
        if len(ids) == 0:
            out.append([]); sizes.append(0); continue
        take = int(min(k, len(ids)))
        w = cnt[ids]
        s = w.sum()
        p = (w / s) if s > 0 else None
        sel = rng.choice(len(ids), size=take, replace=False, p=p)
        chosen = [int(ids[j]) for j in sel]
        out.append(chosen)
        sizes.append(take)
        n_tot += take
        n_liked += sum(1 for i in chosen if int(d.get(i, -1)) >= LIKE_MIN_LEVEL)
    sizes = np.asarray(sizes, np.int64)
    return out, {"k_requested": int(k), "n_users": len(rows),
                 "seed_size_mean": float(sizes.mean()),
                 "seed_size_histogram": {str(int(v)): int(c) for v, c in
                                         zip(*np.unique(sizes, return_counts=True))},
                 "n_users_short_of_k": int((sizes < k).sum()),
                 "frac_volunteered_items_liked": (n_liked / n_tot) if n_tot else None,
                 "draw": "popularity-weighted (p ~ ctx.cnt), without replacement, from the user's "
                         "rated FOLD-IN items (E['lvl'][r], targets excluded by construction)",
                 "rng": f"numpy.default_rng({seed}), users in canonical row order"}


def dedup_keep_order(seq):
    seen, out = set(), []
    for x in seq:
        if x not in seen:
            seen.add(x); out.append(x)
    return out


def split_asked(asked_row, pool):
    """(item ids, concept ids) for one user's realised 16-question interview, in ask order."""
    it = [int(pool[int(j)][1]) for j in asked_row if int(pool[int(j)][0]) == 0]
    cc = [int(pool[int(j)][1]) for j in asked_row if int(pool[int(j)][0]) == 1]
    return it, cc


def build_seqs(asked, pool, vol):
    """The folded sequences for a seeded interview, plus the redundancy count.

    The volunteered items come FIRST (they were volunteered before the interview). An asked item
    already in the volunteered set is counted as redundant and folded ONCE -- a re-ask burns the
    turn and delivers no new evidence, which is exactly what the count is measuring."""
    n = len(vol)
    iseqs, cseqs, redun = [], [], np.zeros(n, np.int64)
    for u in range(n):
        it, cc = split_asked(asked[u], pool)
        vset = set(vol[u])
        redun[u] = sum(1 for i in it if i in vset)
        iseqs.append(dedup_keep_order(list(vol[u]) + it))
        cseqs.append(cc)
    return iseqs, cseqs, redun


# =============================================================================== the seeded rollout
def rollout_seeded(E, pol, phi_t, pool, rows, vol, L=TOTAL_L, batch=1000):
    """The policy's own 16 questions FROM A SEEDED STATE.

    Identical to `policy_net.rollout` in every respect but one: `z_0` is the latent AFTER folding the
    user's volunteered items, so every user starts somewhere different. The volunteered items are NOT
    entered in the asked-history (they were not asked) and are NOT masked out of the candidate pool --
    the policy is free to re-ask one, and how often it does is the redundancy measurement.
    """
    from strategy_channel_suite import genv_eval_multi
    from ndcg_tree import branch_of_per_user
    n, P = len(rows), len(pool)
    asked = np.full((n, L), -1, np.int64)
    outs = np.full((n, L), -1, np.int8)
    asked_mask = np.zeros((n, P), bool)
    empty = [[] for _ in range(n)]
    r0 = genv_eval_multi(E, rows, vol, empty, score=False, return_z=True)
    z = r0[-1].numpy()
    phi_g = torch.from_numpy(phi_t)
    pol.eval()
    curves = {}
    for t in range(L):
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
        iseq, cseq, _ = build_seqs(asked[:, :t + 1], pool, vol)
        r = genv_eval_multi(E, rows, iseq, cseq, return_z=(t + 1 < L), score=need)
        if need:
            curves[L] = (r[0], r[1])
        if t + 1 < L:
            z = r[-1].numpy()
        log(f"    [rollout k-seeded] turn {t + 1}/{L} done")
    assert (asked >= 0).all() and (outs >= 0).all()
    return {"asked": asked, "outs": outs, "curves": curves}


# =============================================================================== per-cell persistence
def cell_path(k, arm):
    return os.path.join(RESUME_DIR, f"resume_k{k}_{arm}.json")


def write_cell(k, arm, blob, per_user):
    os.makedirs(RESUME_DIR, exist_ok=True)
    p = cell_path(k, arm)
    np.savez_compressed(p.replace(".json", "_per_user.npz"), **per_user)
    with open(p + ".tmp", "w") as fh:
        json.dump(blob, fh, indent=2, default=float)
    os.replace(p + ".tmp", p)
    log(f"  [cell] wrote {os.path.basename(p)}")


def read_cell(k, arm, sig):
    p = cell_path(k, arm)
    if not os.path.exists(p):
        return None
    b = json.load(open(p))
    if b.get("contract") != sig:
        log(f"  [cell] {os.path.basename(p)} exists under a DIFFERENT contract -- refusing to resume")
        return None
    z = np.load(p.replace(".json", "_per_user.npz"))
    return b, {kk: z[kk] for kk in z.files}


# =============================================================================== the run
def main():
    ap = argparse.ArgumentParser(description="Experiment R (Part H.2): resumption from seeded states")
    ap.add_argument("--ks", default="2,4,8")
    ap.add_argument("--ckpt", default=os.path.join(OUTDIR, "policy_main_equivariant_both.pt"))
    ap.add_argument("--threads", type=int, default=None, help="consumed by strategy_channel_suite")
    ap.add_argument("--full_threads", action="store_true")
    ap.add_argument("--i_have_author_goahead", action="store_true")
    args = ap.parse_args()
    assert args.i_have_author_goahead, "REFUSING TO RUN without --i_have_author_goahead"
    ks = [int(x) for x in args.ks.split(",") if x.strip()]
    t00 = time.time()
    os.makedirs(RESUME_DIR, exist_ok=True)

    from strategy_channel_suite import (build_greedy_env, contract_selftest, genv_eval, genv_mean,
                                        genv_eval_multi, CERT_TOWER, CERT_CONC)
    from ndcg_tree import build_pool, leak_assert, branch_of_selftest
    torch.manual_seed(SEED); np.random.seed(SEED)

    log("#" * 110)
    log("# EXPERIMENT R -- RESUMPTION (design sheet Part H.2). VALIDATION ONLY.")
    log("# The 10,000 canonical TEST users are NOT built, NOT loaded, NOT touched.")
    log("#" * 110)

    E = build_greedy_env(CERT_TOWER, "validation", CERT_CONC, "resume")
    contract_selftest(E, E["rows"])
    leak_assert(E, E["rows"], "resume/val")
    rows = E["rows"]
    assert len(rows) == 10000, f"validation cohort is {len(rows)} users, expected 10,000"

    # ---- gates before anything is trusted --------------------------------------------------------
    f0, t0_ = genv_mean(E, [], rows)
    assert round(f0, 8) == VAL_M0_FULL, f"CANONICAL SNAP FAILED: {f0!r} != {VAL_M0_FULL}"
    log(f"[gate] canonical m=0 snap full={f0!r} tail={t0_!r} -- exact")

    Dva = load_traj(train=False)
    pool = Dva["pool"]
    pool_e, _ip, _cp = build_pool(E, 500, 500)
    assert pool_e == pool, "the env's pool is not the trajectory bank's pool"
    assert list(np.asarray(Dva["uid"], np.int64)) == list(np.asarray(rows, np.int64)), \
        "the validation trajectory rows are not the env's rows, in order"
    branch_of_selftest(E, rows[:2000], [pool[0], pool[500], (1, 114)])

    static_seq = [(int(a), int(b)) for a, b in json.load(open(STATIC_JSON))["sequence"]]
    sf, stl = genv_eval(E, static_seq, rows)
    assert abs(float(np.nanmean(sf)) - STATIC_VAL_FULL) < 1e-12, "BANKED-HEADLINE ASSERT FAILED"
    log(f"[gate] greedy_static_policyB_combined reproduces EXACTLY: full={float(np.nanmean(sf))!r}")

    z1 = np.load(E1_PER_USER)
    e1_full = np.asarray(z1["policy_full"], np.float64)
    e1_tail = np.asarray(z1["policy_tail"], np.float64)
    e1_asked = np.asarray(z1["asked"], np.int64)
    assert abs(float(e1_full.mean()) - E1_VAL_FULL) < 1e-6, "the banked E1 per-user vector moved"
    assert np.array_equal(e1_asked, np.asarray(Dva["label_idx"], np.int64)), \
        "E1's banked rollout is not the tree's realised interview -- it was, to 1e-9, in E1"
    log(f"[gate] E1 == the tree: banked per-user mean full {float(e1_full.mean()):.9f}")

    # ---- the policy ------------------------------------------------------------------------------
    blob = torch.load(args.ckpt, map_location="cpu")
    assert [(int(a), int(b)) for a, b in blob["pool"]] == pool, "the ckpt was fitted on another pool"
    phi_t, norm = build_phi(E, pool)
    assert np.array_equal(blob["norm_mu"], norm[0]) and np.array_equal(blob["norm_sd"], norm[1]), \
        "the ckpt's descriptor standardisation differs from this env's"
    pol = Policy(E["ctx"].d, phi_t.shape[1], len(pool), arch=blob["arch"], state=blob["state"],
                 use_pos=blob["use_pos"])
    pol.load_state_dict(blob["model"])
    log(f"[policy] {os.path.basename(args.ckpt)} epoch {blob['epoch']} val_full {blob['val_full']:.6f}"
        f" -- FROZEN, no gradient step is taken anywhere in this file")

    sig = {"tower": E["tower"], "conc_ckpt": E["conc_ckpt"], "pool_contract": E["pool_contract"],
           "ckpt": os.path.basename(args.ckpt), "seed": SEED, "L": TOTAL_L,
           "draw": "popularity-weighted, without replacement, from rated fold-in items"}

    tree_asked = np.asarray(Dva["label_idx"], np.int64)
    tree_outs = np.asarray(Dva["outcome"], np.int64)
    static_idx = np.asarray([pool.index(q) for q in static_seq], np.int64)
    static_asked = np.tile(static_idx[None, :], (len(rows), 1))

    results = {}
    for k in ks:
        log("=" * 100)
        log(f"=== k = {k} volunteered items")
        vol, draw_info = draw_volunteered(E, rows, k)
        log(f"[draw] {json.dumps(draw_info, default=float)}")
        cells = {}
        for arm in ARMS:
            got = read_cell(k, arm, sig)
            if got is not None:
                log(f"  [resume] k={k} {arm} already banked -- reusing")
                cells[arm] = got
                continue
            t0 = time.time()
            if arm == "none":
                empty = [[] for _ in rows]
                f, t = genv_eval_multi(E, rows, vol, empty)
                asked = np.full((len(rows), 0), -1, np.int64)
                outs = np.full((len(rows), 0), -1, np.int8)
                redun = np.zeros(len(rows), np.int64)
            elif arm in ("tree", "static"):
                asked = tree_asked if arm == "tree" else static_asked
                iseq, cseq, redun = build_seqs(asked, pool, vol)
                f, t = genv_eval_multi(E, rows, iseq, cseq)
                if arm == "tree":
                    # the tree's ANSWERS cannot move either: the routing rule reads the user's own
                    # ratings, not what they volunteered. Asserted, not assumed.
                    from ndcg_tree import branch_of_per_user
                    o0 = branch_of_per_user(E, rows, [pool[int(j)] for j in asked[:, 0]])
                    assert np.array_equal(np.asarray(o0, np.int64), tree_outs[:, 0]), \
                        "the tree's own ternary answer moved under seeding -- impossible; harness bug"
                    outs = tree_outs
                else:
                    from ndcg_tree import branch_of_per_user
                    outs = np.stack([branch_of_per_user(E, rows, [pool[int(j)] for j in asked[:, tt]])
                                     for tt in range(TOTAL_L)], 1).astype(np.int64)
            else:
                ro = rollout_seeded(E, pol, phi_t, pool, rows, vol)
                asked, outs = ro["asked"], ro["outs"].astype(np.int64)
                f, t = ro["curves"][TOTAL_L]
                _i, _c, redun = build_seqs(asked, pool, vol)
            blob_c = {
                "analysis": "policy_resume_cell", "design_sheet":
                "docs/design/PAPER_B_POLICY_DESIGN.md H.2", "test_touched": False,
                "split": "validation", "k": k, "arm": arm, "contract": sig, "draw": draw_info,
                "full": float(np.nanmean(f)), "tail": float(np.nanmean(t)),
                "n_questions_asked": int(asked.shape[1]),
                "redundant_reasks_mean": float(redun.mean()),
                "redundant_reasks_total": int(redun.sum()),
                "frac_users_with_a_redundant_reask": float((redun > 0).mean()),
                "answer_rate": float((outs != PN.OUTCOME["noanswer"]).mean()) if outs.shape[1] else None,
                "concept_share": channel_mix(asked, pool) if asked.shape[1] else None,
                "n_distinct_interviews": len({tuple(r) for r in asked}) if asked.shape[1] else 1,
                "minutes": round((time.time() - t0) / 60, 2)}
            pu = {"full": f, "tail": t, "asked": asked, "outs": outs, "redundant": redun,
                  "seed_sizes": np.asarray([len(v) for v in vol], np.int64),
                  "rows": np.asarray(rows, np.int64)}
            write_cell(k, arm, blob_c, pu)
            cells[arm] = (blob_c, pu)
            log(f"  [k={k} {arm}] full={blob_c['full']:.5f} tail={blob_c['tail']:.5f} "
                f"redundant={blob_c['redundant_reasks_mean']:.4f} ({blob_c['minutes']:.1f}m)")

        # ---- the comparisons ---------------------------------------------------------------------
        gf = {a: np.asarray(cells[a][1]["full"], np.float64) for a in ARMS}
        gt = {a: np.asarray(cells[a][1]["tail"], np.float64) for a in ARMS}
        pa = np.asarray(cells["policy"][1]["asked"], np.int64)
        diff = (pa != e1_asked)
        adapt = {
            "frac_never_deviates_from_its_own_unseeded_rollout": float((~diff.any(1)).mean()),
            "mean_positions_differing": float(diff.sum(1).mean()),
            "frac_users_differing_at_each_position": [float(diff[:, tt].mean())
                                                      for tt in range(TOTAL_L)],
            "n_positions_differing_histogram": [int((diff.sum(1) == m).sum())
                                                for m in range(TOTAL_L + 1)],
            "note": ("E1's unseeded rollout IS the tree's realised interview (agreement 1.0000). So "
                     "this is simultaneously 'does the policy adapt to the seeding' and 'does it "
                     "leave the teacher's manifold'.")}
        comp = {
            "policy_minus_tree": {"full": paired_bootstrap(gf["policy"] - gf["tree"]),
                                  "tail": paired_bootstrap(gt["policy"] - gt["tree"])},
            "policy_minus_static": {"full": paired_bootstrap(gf["policy"] - gf["static"]),
                                    "tail": paired_bootstrap(gt["policy"] - gt["static"])},
            "tree_minus_static": {"full": paired_bootstrap(gf["tree"] - gf["static"]),
                                  "tail": paired_bootstrap(gt["tree"] - gt["static"])},
            "policy_minus_none": {"full": paired_bootstrap(gf["policy"] - gf["none"]),
                                  "tail": paired_bootstrap(gt["policy"] - gt["none"])},
            "tree_minus_none": {"full": paired_bootstrap(gf["tree"] - gf["none"]),
                                "tail": paired_bootstrap(gt["tree"] - gt["none"])},
            "static_minus_none": {"full": paired_bootstrap(gf["static"] - gf["none"]),
                                  "tail": paired_bootstrap(gt["static"] - gt["none"])},
            "policy_seeded_minus_E1_unseeded": {"full": paired_bootstrap(gf["policy"] - e1_full),
                                                "tail": paired_bootstrap(gt["policy"] - e1_tail)},
            "tree_seeded_minus_E1_unseeded": {"full": paired_bootstrap(gf["tree"] - e1_full),
                                              "tail": paired_bootstrap(gt["tree"] - e1_tail)}}
        results[str(k)] = {"draw": draw_info, "arms": {a: cells[a][0] for a in ARMS},
                           "adaptation": adapt, "comparisons": comp,
                           "pep_divergence_policy": pep_divergence_from_arrays(
                               pa, np.asarray(cells["policy"][1]["outs"], np.int64))}
        p = os.path.join(RESUME_DIR, "policy_resume.json")
        out = {"analysis": "policy_resume_H2", "design_sheet":
               "docs/design/PAPER_B_POLICY_DESIGN.md H.2", "citable": True, "test_touched": False,
               "split": "validation", "n_users": len(rows), "contract": sig,
               "unseeded_reference": {"E1_tree_full": float(e1_full.mean()),
                                      "E1_tree_tail": float(e1_tail.mean()),
                                      "static_full": float(np.nanmean(sf)),
                                      "static_tail": float(np.nanmean(stl)),
                                      "m0_full": f0, "m0_tail": t0_,
                                      "bank_alone_full": STATIC_VAL_FULL - STATIC_ITEMS_VAL_FULL,
                                      "bank_alone_tail": STATIC_VAL_TAIL - STATIC_ITEMS_VAL_TAIL},
               "by_k": results, "minutes": round((time.time() - t00) / 60, 2)}
        with open(p + ".tmp", "w") as fh:
            json.dump(out, fh, indent=2, default=float)
        os.replace(p + ".tmp", p)
        log(f"[k={k}] policy-tree full {comp['policy_minus_tree']['full']['mean']:+.5f} "
            f"[{comp['policy_minus_tree']['full']['lo']:+.5f}, "
            f"{comp['policy_minus_tree']['full']['hi']:+.5f}] | "
            f"policy adapts on {(1 - adapt['frac_never_deviates_from_its_own_unseeded_rollout']) * 100:.2f}% "
            f"of users, mean {adapt['mean_positions_differing']:.3f} positions changed")
    log(f"[done] {round((time.time() - t00) / 60, 2)} minutes")
    return out


if __name__ == "__main__":
    main()
