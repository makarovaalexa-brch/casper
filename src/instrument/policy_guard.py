r"""EXPERIMENT G -- THE GUARD: "new vocab means knowing what NOT to ask".

Design sheet: `docs/design/PAPER_B_POLICY_DESIGN.md` **PART H.4 is binding**. Committed before the
run per HARD RULE 10. Validation only; the 10,000 canonical TEST users are never built, never
loaded, never named.

THE REFRAME, FORCED BY MEASUREMENT (H.4)
----------------------------------------
Catalogue-wide questions are worth about nothing on their own (-0.00104...+0.00119 per draw against
+0.00101...+0.00470 in-pool). So the honest open-vocabulary claim is NOT "it finds treasure outside
the pool" -- the treasure mostly is not there -- but **"it can score any question and correctly
declines the junk."** E1 opened to all 19,390 candidates costs **-0.02034 [-0.02236, -0.01837]** of
full NDCG@10, and the mechanism is measured: its out-of-pool picks are answered **9.1%** of the time
against 55.2% for in-pool picks. A burned turn costs the interview a question.

THE ARMS
--------
  pool          E1 restricted to its 1,000-candidate training pool. THE CEILING (0.28355 banked).
  open          E1 open to all 19,390, unguarded. THE DEFICIT (0.26321 banked, -0.02034).
  guard@tau     open vocabulary, with the frozen REACTION HEAD as a HARD guard: a candidate whose
                predicted P(no-answer | state, q) exceeds `tau` is masked out of the argmax. The
                guard is STATE-CONDITIONED -- it re-scores every candidate at every turn.
  dumb@m        ★ THE COMPARATOR THAT DECIDES WHETHER THE LEARNED HEAD EARNS ANYTHING. The same
                open vocabulary, guarded by the state-free per-question EMPIRICAL answer rate
                (the popularity prior banked with the outcome head, computed on train users). It is
                BUDGET-MATCHED: it keeps exactly as many candidates as the learned guard left alive
                on average at that threshold, so the two guards differ only in whether the guard
                reads the interview. If popularity alone recovers the same deficit, the learned head
                adds nothing and that is what gets written.

THE SWEEP IS DECLARED, NOT TUNED. Every threshold is reported -- the whole curve -- and no point is
picked post hoc. `tau = 1.0` is the unguarded arm by construction, which makes the curve's left end
a free reproduction check against the banked open-vocabulary number.

FALSIFIER, PRE-REGISTERED
-------------------------
If recovery requires driving the out-of-pool pick rate to ~0%, the claim collapses to **"the pool was
already right"** -- and that is reported, because it is informative about the CATALOGUE, not only
about the model. Success is the opposite shape: most of the -0.02034 recovered while a NONZERO
out-of-pool rate survives, with those picks answered at a rate near the in-pool rate.

WHAT MAY NOT BE WRITTEN (the rule carried from Experiment R): if the guard fires structurally but
produces no behavioural difference, say so in those words. A capability that changes no behaviour is
not a result.
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

from policy_net import (OUTDIR, SEED, TOTAL_L, OUTCOME, Policy, build_phi, paired_bootstrap,
                        channel_mix, log, VAL_M0_FULL, E1_PER_USER)
from outcome_head import OutcomeModel, pool_to_catalogue

GDIR = os.path.join(OUTDIR, "vocab_guard")
# banked, asserted rather than trusted (`banked-global-greedy-is-a-policy-A-artifact`)
E1_POOL_FULL = 0.28354971825495506
E1_OPEN_FULL = 0.26320921379832823
E1_OPEN_TAIL = 0.12180505138899793
E1_OPEN_OOP_RATE = 0.299925
E1_OPEN_ANSWER_RATE = 0.41394375


def rollout_open(E, pol, head, phi_pol, phi_head, full_pool, rows, guard=None, tau=None,
                 keep_mask=None, batch=250, L=TOTAL_L):
    """A 16-question rollout over the OPEN vocabulary, with an optional hard guard.

      guard=None      unguarded (E1's scalability arm, reproduced)
      guard='head'    mask candidates whose predicted P(no-answer | state, q) > `tau`. Recomputed at
                      every turn from the CURRENT state -- this is the state-conditioned guard.
      guard='dumb'    mask everything outside the fixed `keep_mask` (the budget-matched, state-free
                      popularity/answer-rate guard).

    A guard that masks EVERY candidate at some state would make the argmax undefined; in that case
    the guard is released for that user at that turn and the release is COUNTED and reported, never
    silently absorbed.
    """
    from strategy_channel_suite import genv_eval_multi
    from ndcg_tree import branch_of_per_user
    n, P = len(rows), len(full_pool)
    asked = np.full((n, L), -1, np.int64)
    outs = np.full((n, L), -1, np.int8)
    asked_mask = np.zeros((n, P), bool)
    empty = [[] for _ in range(n)]
    r0 = genv_eval_multi(E, rows, empty, empty, score=False, return_z=True)
    z = r0[-1].numpy()
    n_released = 0
    n_masked_tot = 0.0
    phi_pol_g = torch.from_numpy(phi_pol)
    phi_head_g = torch.from_numpy(phi_head)
    pol.eval()
    kv = None
    if guard == "head":
        with torch.no_grad():
            kv = head.head.keys(phi_head_g)
    for t in range(L):
        hq = np.zeros((n, TOTAL_L - 1), np.int64); ho = np.zeros((n, TOTAL_L - 1), np.int64)
        hm = np.zeros((n, TOTAL_L - 1), np.float32)
        for j in range(t):
            hq[:, j] = asked[:, j]; ho[:, j] = outs[:, j]; hm[:, j] = 1.0
        pos = np.tile(np.asarray([[t, TOTAL_L - t]], np.float32), (n, 1))
        with torch.no_grad():
            for st in range(0, n, batch):
                sl = slice(st, min(st + batch, n))
                zb = torch.from_numpy(z[sl]); hob = torch.from_numpy(ho[sl])
                hmb = torch.from_numpy(hm[sl]); pb = torch.from_numpy(pos[sl])
                am = torch.from_numpy(asked_mask[sl])
                s = pol(zb, torch.from_numpy(phi_pol[hq[sl]]), hob, hmb, pb, phi_pol_g, am)
                if guard == "head":
                    lg = head(zb, torch.from_numpy(phi_head[hq[sl]]), hob, hmb, pb, kv=kv)
                    pna = torch.softmax(lg, 0)[OUTCOME["noanswer"]]          # (B,P)
                    block = pna > tau
                elif guard == "dumb":
                    block = torch.from_numpy(~keep_mask)[None, :].expand(s.shape[0], -1)
                else:
                    block = None
                if block is not None:
                    block = block & ~am                       # already-asked are handled by `pol`
                    alive = (~(block | am)).sum(1)
                    dead = (alive == 0)
                    n_released += int(dead.sum())
                    n_masked_tot += float(block.sum())
                    if dead.any():
                        block[dead] = False                   # release rather than crash
                    s = s.masked_fill(block, float("-inf"))
                asked[sl, t] = s.argmax(1).numpy()
        outs[:, t] = branch_of_per_user(E, rows, [full_pool[int(j)] for j in asked[:, t]])
        asked_mask[np.arange(n), asked[:, t]] = True
        need = (t + 1 == L)
        iseq = [[full_pool[int(j)][1] for j in asked[u, :t + 1] if full_pool[int(j)][0] == 0]
                for u in range(n)]
        cseq = [[full_pool[int(j)][1] for j in asked[u, :t + 1] if full_pool[int(j)][0] == 1]
                for u in range(n)]
        r = genv_eval_multi(E, rows, iseq, cseq, return_z=(t + 1 < L), score=need)
        if need:
            f, tl = r[0], r[1]
        if t + 1 < L:
            z = r[-1].numpy()
        log(f"    [turn {t + 1}/{L}] done")
    assert (asked >= 0).all() and (outs >= 0).all()
    return {"asked": asked, "outs": outs, "full": f, "tail": tl,
            "n_guard_releases": int(n_released),
            "mean_candidates_masked_per_state": n_masked_tot / max(n * L, 1)}


def describe(asked, outs, full_pool, trained, f, tl):
    from collections import Counter
    oop = np.asarray([[0 if full_pool[int(j)] in trained else 1 for j in r] for r in asked], bool)
    ans = (outs != OUTCOME["noanswer"])
    c = Counter(full_pool[int(j)] for j in asked.ravel() if full_pool[int(j)] not in trained)
    return {"full": float(np.nanmean(f)), "tail": float(np.nanmean(tl)),
            "out_of_pool_pick_rate": float(oop.mean()),
            "out_of_pool_by_position": [float(x) for x in oop.mean(0)],
            "users_with_any_out_of_pool_pick": float((oop.sum(1) > 0).mean()),
            "n_distinct_out_of_pool_questions": len(c),
            "answer_rate": float(ans.mean()),
            "answer_rate_of_in_pool_picks": float(ans[~oop].mean()) if (~oop).any() else None,
            "answer_rate_of_out_of_pool_picks": float(ans[oop].mean()) if oop.any() else None,
            "concept_share": channel_mix(asked, full_pool),
            "n_distinct_interviews": len({tuple(r) for r in asked})}


def main():
    ap = argparse.ArgumentParser(description="Experiment G (Part H.4): the open-vocabulary guard")
    ap.add_argument("--taus", default="1.0,0.999,0.99,0.98,0.95,0.90,0.80,0.50")
    ap.add_argument("--ckpt", default=os.path.join(OUTDIR, "policy_main_equivariant_both.pt"))
    ap.add_argument("--head_ckpt", default=os.path.join(OUTDIR, "outcome_head_main_both.pt"))
    ap.add_argument("--pop_npz", default=os.path.join(OUTDIR, "outcome_head_main_both_pop.npz"))
    ap.add_argument("--batch", type=int, default=250)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--full_threads", action="store_true")
    ap.add_argument("--no_dumb", action="store_true", help="skip the budget-matched popularity guard")
    ap.add_argument("--i_have_author_goahead", action="store_true")
    args = ap.parse_args()
    assert args.i_have_author_goahead, "REFUSING TO RUN without --i_have_author_goahead"
    taus = [float(x) for x in args.taus.split(",") if x.strip()]
    os.makedirs(GDIR, exist_ok=True)
    torch.manual_seed(SEED); np.random.seed(SEED)
    t00 = time.time()

    from strategy_channel_suite import (build_greedy_env, contract_selftest, genv_mean,
                                        CERT_TOWER, CERT_CONC)
    from ndcg_tree import build_pool, leak_assert, branch_of_selftest

    log("#" * 110)
    log("# EXPERIMENT G -- THE OPEN-VOCABULARY GUARD (design sheet Part H.4). VALIDATION ONLY.")
    log("# The 10,000 canonical TEST users are NOT built, NOT loaded, NOT touched.")
    log("#" * 110)

    E = build_greedy_env(CERT_TOWER, "validation", CERT_CONC, "guard")
    contract_selftest(E, E["rows"]); leak_assert(E, E["rows"], "guard/val")
    rows = E["rows"]
    assert len(rows) == 10000
    f0, t0_ = genv_mean(E, [], rows)
    assert round(f0, 8) == VAL_M0_FULL, f"CANONICAL SNAP FAILED: {f0!r}"
    log(f"[gate] canonical m=0 snap full={f0!r} -- exact")

    pool, _ip, _cp = build_pool(E, 500, 500)
    branch_of_selftest(E, rows[:2000], [pool[0], pool[500], (1, 114)])
    ni, C = E["ctx"].ni, E["C"]
    full_pool = [(0, i) for i in range(ni)] + [(1, c) for c in range(C)]
    P = len(full_pool)
    trained = set(pool)
    log(f"[vocab] training pool {len(pool)} -> open catalogue {P} ({ni} items + {C} concepts)")

    # ---- the two frozen models, each with ITS OWN descriptor standardisation --------------------
    blob = torch.load(args.ckpt, map_location="cpu")
    assert [(int(a), int(b)) for a, b in blob["pool"]] == pool, "the policy ckpt is on another pool"
    phi_pol, _ = build_phi(E, full_pool, norm=(blob["norm_mu"], blob["norm_sd"]))
    pol = Policy(E["ctx"].d, phi_pol.shape[1], P, arch=blob["arch"], state=blob["state"],
                 use_pos=blob["use_pos"])
    pol.load_state_dict(blob["model"])

    hb = torch.load(args.head_ckpt, map_location="cpu")
    phi_head, hnorm = build_phi(E, full_pool, norm=(hb["norm_mu"], hb["norm_sd"]))
    assert int(hb["d_phi"]) == phi_head.shape[1] and int(hb["d_z"]) == E["ctx"].d
    head = OutcomeModel(int(hb["d_z"]), int(hb["d_phi"]), state=hb["state"], use_pos=hb["use_pos"],
                        d_k=int(hb["d_k"]), d_h=int(hb["d_h"]))
    head.load_state_dict(hb["model"]); head.eval()
    assert not hb.get("prior_feat"), "the guard must not use a per-question prior feature"
    log(f"[models] policy {os.path.basename(args.ckpt)} (epoch {blob['epoch']}) + reaction head "
        f"{os.path.basename(args.head_ckpt)} (epoch {hb['epoch']}) -- BOTH FROZEN. The value model "
        f"is DELETED: there is no lift head and no P*V term anywhere in this file.")

    # ---- the dumb guard's score: the state-free per-question empirical answer rate ---------------
    pz = np.load(args.pop_npz)
    pop = np.asarray(pz["pop"], np.float64)                    # (P,3) over the catalogue ordering
    assert pop.shape[0] == P, f"the banked prior is {pop.shape[0]} wide, the catalogue is {P}"
    dumb_score = 1.0 - pop[:, OUTCOME["noanswer"]]             # empirical P(answer), no state
    log(f"[dumb] state-free per-question answer rate: mean {dumb_score.mean():.5f}, "
        f"max {dumb_score.max():.5f}; {int((dumb_score > 0.5).sum())} candidates above 0.5")

    z1 = np.load(E1_PER_USER)
    e1_pool_full = np.asarray(z1["policy_full"], np.float64)
    e1_pool_tail = np.asarray(z1["policy_tail"], np.float64)
    assert abs(float(np.nanmean(e1_pool_full)) - E1_POOL_FULL) < 1e-9, "banked E1 pool vector moved"

    sig = {"tower": E["tower"], "conc_ckpt": E["conc_ckpt"], "pool_contract": E["pool_contract"],
           "policy_ckpt": os.path.basename(args.ckpt), "head_ckpt": os.path.basename(args.head_ckpt),
           "seed": SEED, "L": TOTAL_L, "n_catalogue": P}

    def cell(name, fn):
        p = os.path.join(GDIR, f"{name}.json")
        if os.path.exists(p):
            b = json.load(open(p))
            if b.get("contract") == sig:
                z = np.load(p.replace(".json", "_per_user.npz"))
                log(f"  [resume] {name} already banked -- reusing")
                return b, {k: z[k] for k in z.files}
            log(f"  [resume] {name} exists under a DIFFERENT contract -- refusing to resume")
        t0 = time.time()
        ro = fn()
        d = describe(ro["asked"], ro["outs"], full_pool, trained, ro["full"], ro["tail"])
        b = {"analysis": "policy_guard_cell", "design_sheet":
             "docs/design/PAPER_B_POLICY_DESIGN.md H.4", "test_touched": False,
             "split": "validation", "cell": name, "contract": sig,
             "n_guard_releases": ro["n_guard_releases"],
             "mean_candidates_masked_per_state": ro["mean_candidates_masked_per_state"],
             "mean_candidates_alive_per_state": P - ro["mean_candidates_masked_per_state"],
             "minutes": round((time.time() - t0) / 60, 2), **d}
        pu = {"full": ro["full"], "tail": ro["tail"], "asked": ro["asked"], "outs": ro["outs"],
              "rows": np.asarray(rows, np.int64)}
        np.savez_compressed(p.replace(".json", "_per_user.npz"), **pu)
        with open(p + ".tmp", "w") as fh:
            json.dump(b, fh, indent=2, default=float)
        os.replace(p + ".tmp", p)
        log(f"  [cell] {name}: full {b['full']:.5f} tail {b['tail']:.5f} | oop {b['out_of_pool_pick_rate']*100:.1f}% "
            f"| answer {b['answer_rate']:.4f} (oop {b['answer_rate_of_out_of_pool_picks']}) "
            f"| alive/state {b['mean_candidates_alive_per_state']:.0f} ({b['minutes']:.1f}m)")
        return b, pu

    cells = {}
    # the CEILING: E1 restricted to its own pool. Reproduced, not quoted.
    phi_small, _ = build_phi(E, pool, norm=(blob["norm_mu"], blob["norm_sd"]))
    pol_small = Policy(E["ctx"].d, phi_small.shape[1], len(pool), arch=blob["arch"],
                       state=blob["state"], use_pos=blob["use_pos"])
    pol_small.load_state_dict(blob["model"])
    log("=== arm: POOL (the ceiling)")
    from policy_net import rollout as e1_rollout
    p2c = pool_to_catalogue(pool, ni)          # pool index -> catalogue index, so every arm's
                                               # `asked` lives in ONE index space and `describe`
                                               # cannot silently read the wrong question
    def _pool():
        r = e1_rollout(E, pol_small, phi_small, pool, rows)
        f, tl = r["curves"][TOTAL_L]
        return {"asked": p2c[r["asked"]], "outs": r["outs"], "full": f, "tail": tl,
                "n_guard_releases": 0, "mean_candidates_masked_per_state": float(P - len(pool))}
    b_pool, pu_pool = cell("pool", _pool)
    assert abs(b_pool["full"] - E1_POOL_FULL) < 1e-9, (
        f"BANKED-HEADLINE ASSERT FAILED: the pool-restricted arm rescores to {b_pool['full']!r}, "
        f"banked {E1_POOL_FULL}")
    log(f"[gate] pool-restricted arm reproduces the banked E1 number EXACTLY: {b_pool['full']!r}")
    cells["pool"] = (b_pool, pu_pool)

    for tau in taus:
        name = "open_unguarded" if tau >= 1.0 else f"guard_tau{tau:g}"
        log(f"=== arm: {name}")
        if tau >= 1.0:
            # the unguarded arm goes through E1's OWN rollout, unchanged, so the reproduction check
            # below is a check on the banked number and not on a lookalike code path
            def _open():
                r = e1_rollout(E, pol, phi_pol, full_pool, rows)
                f, tl = r["curves"][TOTAL_L]
                return {"asked": r["asked"], "outs": r["outs"], "full": f, "tail": tl,
                        "n_guard_releases": 0, "mean_candidates_masked_per_state": 0.0}
            cells[name] = cell(name, _open)
        else:
            cells[name] = cell(name, lambda tau=tau: rollout_open(
                E, pol, head, phi_pol, phi_head, full_pool, rows,
                guard="head", tau=tau, batch=args.batch))
        if tau >= 1.0:
            d = cells[name][0]
            log(f"[gate] unguarded open vocabulary: full {d['full']:.8f} (banked {E1_OPEN_FULL}), "
                f"oop {d['out_of_pool_pick_rate']:.6f} (banked {E1_OPEN_OOP_RATE})")
            assert abs(d["full"] - E1_OPEN_FULL) < 1e-9, (
                f"BANKED-HEADLINE ASSERT FAILED: the unguarded open arm rescores to {d['full']!r}, "
                f"banked {E1_OPEN_FULL}")

    # ---- the BUDGET-MATCHED dumb guard ----------------------------------------------------------
    if not args.no_dumb:
        order = np.argsort(-dumb_score, kind="stable")
        for tau in taus:
            if tau >= 1.0:
                continue
            src = cells[f"guard_tau{tau:g}"][0]
            m = int(round(src["mean_candidates_alive_per_state"]))
            m = int(np.clip(m, 1, P))
            keep = np.zeros(P, bool); keep[order[:m]] = True
            name = f"dumb_m{m}"
            if name in cells:
                continue
            log(f"=== arm: {name} (budget-matched to guard_tau{tau:g}: {m} candidates alive)")
            cells[name] = cell(name, lambda keep=keep: rollout_open(
                E, pol, head, phi_pol, phi_head, full_pool, rows,
                guard="dumb", keep_mask=keep, batch=args.batch))
            cells[name][0]["matched_to"] = f"guard_tau{tau:g}"

    # ---- the comparisons ------------------------------------------------------------------------
    base = cells["open_unguarded"][1] if "open_unguarded" in cells else None
    out = {"analysis": "policy_guard_H4", "design_sheet":
           "docs/design/PAPER_B_POLICY_DESIGN.md H.4", "citable": True, "test_touched": False,
           "split": "validation", "n_users": len(rows), "contract": sig,
           "banked": {"E1_pool_full": E1_POOL_FULL, "E1_open_full": E1_OPEN_FULL,
                      "E1_open_deficit_full": E1_OPEN_FULL - E1_POOL_FULL,
                      "E1_open_oop_rate": E1_OPEN_OOP_RATE,
                      "E1_open_answer_rate": E1_OPEN_ANSWER_RATE},
           "dumb_guard": "state-free per-question empirical answer rate, budget-matched by the "
                         "number of candidates the learned guard leaves alive",
           "arms": {}, "minutes": None}
    for name, (b, pu) in cells.items():
        r = dict(b)
        if base is not None and name != "open_unguarded":
            r["vs_open_unguarded"] = {
                "full": paired_bootstrap(np.asarray(pu["full"], np.float64) -
                                         np.asarray(base["full"], np.float64)),
                "tail": paired_bootstrap(np.asarray(pu["tail"], np.float64) -
                                         np.asarray(base["tail"], np.float64))}
        if name != "pool":
            r["vs_pool_ceiling"] = {
                "full": paired_bootstrap(np.asarray(pu["full"], np.float64) - e1_pool_full),
                "tail": paired_bootstrap(np.asarray(pu["tail"], np.float64) - e1_pool_tail)}
        out["arms"][name] = r
    # ★ THE COMPARISON THE PRE-REGISTRATION TURNS ON: learned guard vs the budget-matched dumb one,
    # paired, at every threshold. It belongs in the artifact, not only in the write-up.
    out["learned_minus_dumb_at_matched_budget"] = {}
    for name, (b, pu) in cells.items():
        if not name.startswith("dumb_") or "matched_to" not in b:
            continue
        src = b["matched_to"]
        lp = cells[src][1]
        out["learned_minus_dumb_at_matched_budget"][f"{src}_vs_{name}"] = {
            "candidates_alive": b["mean_candidates_alive_per_state"],
            "learned_full": cells[src][0]["full"], "dumb_full": b["full"],
            "full": paired_bootstrap(np.asarray(lp["full"], np.float64) -
                                     np.asarray(pu["full"], np.float64)),
            "tail": paired_bootstrap(np.asarray(lp["tail"], np.float64) -
                                     np.asarray(pu["tail"], np.float64))}
    out["minutes"] = round((time.time() - t00) / 60, 2)
    p = os.path.join(GDIR, "vocab_guard.json")
    with open(p + ".tmp", "w") as fh:
        json.dump(out, fh, indent=2, default=float)
    os.replace(p + ".tmp", p)
    log(f"[done] {out['minutes']} minutes -> {p}")
    return out


if __name__ == "__main__":
    main()
