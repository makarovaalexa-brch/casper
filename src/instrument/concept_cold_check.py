r"""concept_cold_check.py -- score concept checkpoints with concepts folded CLEAN (zero items).

WHY THIS EXISTS. The in-training val metric handed every user FOUR ITEMS of context and then folded
concepts on top. That is the situation in which concepts matter LEAST, and it introduces a confound: with
items present, member-drop removes the answered concept's member films from the input, so the channel has
to repay a debt that only exists because items were supplied. It is what made m=1 look like a crater.

Concept elicitation is a COLD-START mechanism -- you ask about concepts precisely because you have
nothing else. Folded clean:
  * m=0 is the true no-evidence intercept, so the no-crater constraint is finally within-protocol
  * m=1/4/8 measure what concepts ALONE buy
  * member-drop is vacuous (no items to drop), so the confound disappears

Training needs no redo: the curriculum samples item context k ~ U{0..8}, so pure-concept examples are
already ~11% of it. Only the measurement was wrong, and every epoch is checkpointed.

CANDIDATE POOL (changed 2026-08-05, ruling in external_literature/findings/candidate_masking_protocol.md):
POLICY B -- every rated fold-in-side item of the user (likes AND dislikes, targets excluded) is -inf,
identically for the concept arm and the item arm at every budget. The previous rule masked only the
ANSWERED asked items and applied no fold-in mask at all, which handed the item arm a free lift the
concept arm could not receive by construction (a concept question masks nothing). Everything this file
and `bank_sweep.py` printed before that date is on the old pool and moves -- see
docs/results/CONCEPT_VS_ITEM_BANK_SWEEP_POLICYB.md.

  python src/instrument/concept_cold_check.py --tag cv3 --snapshot models/i26c_ep31/t2i26c_ep31.pt
"""
import os
import sys
import json
import glob
import time
import argparse

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_ROOT, "src", "baselines"))

import concept_curriculum_v2 as CV2
from concept_fold import ConceptFoldNet, fold_items_frozen
import concept_distill_train as CD
from signed_answers import signed_for_pool, cap_negatives, BAND_REFUSE
from i26_encoder import UNSEEN_LEVEL
import mask_contract as MCK

CKPT_DIR = os.path.join(_ROOT, ".cache", "instrument")
MS = (0, 1, 2, 4, 8, 16)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def cold_interviews(S, mc, n_users=1500, seed=4242, policy="broad", rank=None, keep_meh=False):
    """One cold interview per user per budget: NO items, m concepts asked under `policy`.

    `rank`: optional deterministic ranking (top-m asked in order). This is what makes the concept arm
    comparable to the item arm -- items are asked from a RANKED user-independent bank, so drawing
    concepts at random handicaps them. With `rank` set, `policy` is ignored.
    """
    # SEPARATE STREAMS: targets must not depend on how many concepts get drawn, or the concept and
    # item arms end up with different held-out sets and are no longer comparable. Canary: at m=0 both
    # arms fold nothing, so they must score bit-identically.
    rng_t = np.random.default_rng(seed)
    rng = np.random.default_rng(seed + 1)
    users = S["users"][-n_users:]
    packs = {m: [] for m in MS}
    for u in users:
        items = np.asarray(u["items"], np.int64); lv = np.asarray(u["levels"], np.int64)
        liked = np.asarray(u["liked"], np.int64)
        if len(items) < 4 or len(liked) < 2:
            continue
        ntg = max(1, len(liked) // 3)
        tg = rng_t.choice(liked, size=ntg, replace=False)
        keep = ~np.isin(items, tg)
        pool_s, pool_l = items[keep], lv[keep]
        if len(pool_s) < 2:
            continue
        stars = (pool_l.astype(np.float32) + 1.0) / 2.0
        V, B, ans = signed_for_pool(pool_s, stars, S["item_mean"], S["Mm"], S["grate"], S["prereg"],
                                    keep_meh=keep_meh)
        for m in MS:
            if m == 0:
                cids = np.empty(0, np.int64); ok = np.empty(0, bool)
            else:
                cids = (np.asarray(rank[:m], np.int64) if rank is not None
                        else CV2.ask_pool(policy, len(S["tags"]), mc, rng, m))
                ok = np.array([bool(ans[c]) and B[c] != BAND_REFUSE for c in cids])
            cvals = cap_negatives([float(V[c]) if ok[j] else 0.0 for j, c in enumerate(cids)])
            crefs = [0.0 if ok[j] else 1.0 for j in range(len(cids))]
            packs[m].append((np.empty(0, np.int64), np.empty(0, np.int64),      # ZERO ITEMS
                             [int(c) for c in cids], [float(v) for v in cvals],
                             [float(x) for x in crefs], tg, np.empty(0, np.int64),
                             pool_s))                       # [7] POLICY-B candidate exclusion
    return packs


def cold_item_interviews(S, pop_rank, n_users=1500, seed=4242):
    """MATCHED item arm: same users, same targets, same budgets -- but the q questions are ITEMS.

    Realistic deployment contract, mirroring the concept arm question for question:
      * user-independent fixed bank, asked in popularity order (`items-pop`, the strongest item
        strategy in the deployment ledger)
      * an item the user has not rated BURNS the question, exactly as a concept refusal does. It is
        folded as an exposure token (UNSEEN_LEVEL), which is the tower's native "asked, not seen".
      * an asked item that is a held-out TARGET is also folded as unseen -- we must not reveal it.
      * CANDIDATE POOL = POLICY B, identical to the concept arm's (see `score`). It used to be
        "answered items are masked; the concept arm has no items so nothing is masked there" --
        i.e. the item arm got a free lift, worth ~4 top-ranked guaranteed non-targets per user at
        m=16, that the concept arm structurally could not get. The pool is now the same object for
        both arms at every budget, so it cancels in the comparison instead of funding one side.
    """
    rng_t = np.random.default_rng(seed)          # SAME target stream as the concept arm
    users = S["users"][-n_users:]
    packs = {m: [] for m in MS}
    for u in users:
        items = np.asarray(u["items"], np.int64); lv = np.asarray(u["levels"], np.int64)
        liked = np.asarray(u["liked"], np.int64)
        if len(items) < 4 or len(liked) < 2:
            continue
        ntg = max(1, len(liked) // 3)
        tg = rng_t.choice(liked, size=ntg, replace=False)
        keep = ~np.isin(items, tg)
        pool_s, pool_l = items[keep], lv[keep]
        if len(pool_s) < 2:
            continue
        lvl_of = dict(zip(pool_s.tolist(), pool_l.tolist()))
        for m in MS:
            asked = pop_rank[:m]
            ids = np.asarray(asked, np.int64)
            lvs = np.asarray([lvl_of.get(int(i), UNSEEN_LEVEL) for i in asked], np.int64)
            answered = ids[lvs < UNSEEN_LEVEL]
            packs[m].append((ids, lvs, [], [], [], tg, answered,
                             pool_s))                       # [7] POLICY-B candidate exclusion
    return packs


def build_banks(S, n_est=2000, log=log):
    """Rank the concept bank and the item bank by the SAME criterion, so the two arms are comparable.

    Criterion = Shannon entropy of the 4-way outcome of ASKING the question, over held-out training
    users (disjoint from the eval users, which are S["users"][-n_users:]):

        concepts : {like, meh, dislike, REFUSE}      bands straight from signed_for_pool
        items    : {like, meh, dislike, UNSEEN}      like = level>=7 (r>3.5, canonical binarisation),
                                                     dislike = level<=4 (r<=2.5), unseen = not rated

    Popularity ranks how ANSWERABLE a question is; entropy ranks how much it DISCRIMINATES. A blockbuster
    everybody liked is maximally answerable and carries almost no signal.
    """
    est = S["users"][:n_est]
    nC, ni = len(S["tags"]), S["ni"]
    cc = np.zeros((nC, 4), np.float64)
    ci = np.zeros((ni, 4), np.float64)
    n_used = 0
    for u in est:
        items = np.asarray(u["items"], np.int64); lv = np.asarray(u["levels"], np.int64)
        if len(items) < 4:
            continue
        n_used += 1
        stars = (lv.astype(np.float32) + 1.0) / 2.0
        _V, B, _a = signed_for_pool(items, stars, S["item_mean"], S["Mm"], S["grate"], S["prereg"])
        np.add.at(cc, (np.arange(nC), np.asarray(B, np.int64)), 1.0)
        ib = np.full(len(items), 1, np.int64)                  # meh
        ib[lv >= 7] = 0                                        # like   (r > 3.5)
        ib[lv <= 4] = 2                                        # dislike(r <= 2.5)
        np.add.at(ci, (items, ib), 1.0)
    ci[:, 3] = n_used - ci[:, :3].sum(1)                       # unseen = never rated by these users

    def H(c):
        p = c / np.maximum(c.sum(1, keepdims=True), 1e-9)
        return -(p * np.log(np.maximum(p, 1e-12))).sum(1)

    def polar(c, min_support=0.02):
        """DISCRIMINATION, decoupled from answerability: H over {like,meh,dislike} GIVEN answered.

        4-way entropy that includes the unanswerable bucket is DEGENERATE on a sparse catalogue: for
        almost every item the distribution is ~100% 'unseen', so only ultra-popular items score at all
        and the ranking collapses onto popularity (measured: top-16 of the 4-way bank sits at the 99.9th
        popularity percentile and overlaps items-pop 12/16, with identical 59.8% burn). Conditioning on
        'answered' is what actually asks "does this question SPLIT people?".

        Support floor: an item three people answered must not win on noise.
        """
        ans = c[:, :3]
        tot = ans.sum(1)
        h = H(ans)
        h[tot < min_support * max(c.sum(1).max(), 1.0)] = -np.inf
        return h

    def expinfo(c, min_support=0.02):
        """EXPECTED INFORMATION = P(answered) * H(bands | answered).

        The two extremes are both wrong. 4-way entropy collapses onto popularity (answerability
        dominates a sparse catalogue); pure conditional entropy picks maximally divisive questions that
        almost nobody can answer (measured: 5.4% answer rate, so 16 asks return <1 answer). A question is
        worth asking in proportion to how likely it is to be answered TIMES how much the answer splits
        people. This is the defensible static bank.
        """
        tot = c.sum(1)
        p_ans = c[:, :3].sum(1) / np.maximum(tot, 1e-9)
        v = p_ans * H(c[:, :3])
        v[c[:, :3].sum(1) < min_support * max(tot.max(), 1.0)] = -np.inf
        return v

    hc, hi = H(cc), H(ci)
    pc, pi = polar(cc), polar(ci)
    ec, ei = expinfo(cc), expinfo(ci)
    log(f"[banks] estimated on {n_used} held-out training users "
        f"(eval users are the LAST 1500, disjoint)")
    return {"conc_entropy": np.argsort(-hc), "item_entropy": np.argsort(-hi),
            "conc_polar": np.argsort(-pc), "item_polar": np.argsort(-pi),
            "conc_expinfo": np.argsort(-ec), "item_expinfo": np.argsort(-ei),
            "conc_H": hc, "item_H": hi}


def ndcg_at(sc_row, targets, head_mask, k=10):
    """full = NDCG@10 over ALL items. tail = NDCG@10 over the TAIL-ONLY candidate list.

    The tail arm MUST delete head items from the candidate list (`sc[head] = -inf`), matching
    run_battery_phaseA.ndcg10_from_scores: that is the canonical definition and the one every
    published CASPER tail number uses. Scoring tail globally instead -- counting tail targets that
    survive in the unrestricted top-10 -- returns 0.0000 identically at the no-evidence prior,
    because the 10 most popular items fill every slot. That is a metric artifact, not a channel
    property; it is what produced the phantom "tail collapse" of 2026-08-03.
    """
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    order = np.argsort(-sc_row)[:k]
    rel = np.isin(order, targets).astype(np.float64)
    idcg = disc[:min(len(targets), k)].sum()
    full = (rel * disc).sum() / max(idcg, 1e-9)

    t_tg = targets[~head_mask[targets]]
    if len(t_tg) == 0:
        return full, None
    sct = sc_row.copy()
    sct[head_mask] = -np.inf                      # TAIL-ONLY candidate list
    order_t = np.argsort(-sct)[:k]
    rt = np.isin(order_t, t_tg).astype(np.float64)
    it = disc[:min(len(t_tg), k)].sum()
    return full, (rt * disc).sum() / max(it, 1e-9)


def apply_pool_mask(sc_row, e, uid, tag):
    """POLICY B: -inf every rated fold-in-side item of this user (likes AND dislikes), whatever the
    arm asked. e[7] is `pool_s` -- the user's full rated list with their held-out targets removed --
    so targets stay rankable and the mask is the same object for the concept arm and the item arm at
    every budget. The contract assert makes that equality machine-checked rather than asserted in a
    docstring: same user => same mask, same targets, across arms and budgets.

    BEFORE 2026-08-05 this masked only `e[6]` (the answered asked items) with NO fold-in mask at all,
    which is policy C-answered: strategy-dependent, channel-dependent (a concept question masks
    nothing), and a free lift to whichever arm asks items the user has actually seen.
    """
    MCK.assert_fixed_user_mask(tag, uid, e[7], e[5])
    assert not np.isin(e[5], e[7]).any(), "POOL/TARGET LEAK: a held-out target is inside the mask"
    if len(e[7]):
        sc_row[e[7]] = -np.inf
    return sc_row


def score(net, S, packs, head_mask, batch=256, tag="cold_check"):
    Wd, bd = S["Wd"], S["bd"]
    out = {}
    if net is not None:
        net.eval()
    with torch.no_grad():
        for m, rows in packs.items():
            fs, ts = [], []
            for st in range(0, len(rows), batch):
                blk = rows[st:st + batch]
                Zi = fold_items_frozen(S["enc"], [(e[0], e[1]) for e in blk], batch=batch)
                z = Zi if net is None else net(Zi, [e[2] for e in blk], [e[3] for e in blk],
                                               [e[4] for e in blk])
                sc = (z @ Wd.T + bd).numpy()
                for r, e in enumerate(blk):
                    apply_pool_mask(sc[r], e, st + r, tag)
                    f, t = ndcg_at(sc[r], e[5], head_mask)
                    fs.append(f)
                    if t is not None:
                        ts.append(t)
            out[m] = (float(np.mean(fs)), float(np.mean(ts)) if ts else float("nan"))
    if net is not None:
        net.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="cv3")
    ap.add_argument("--snapshot", default=os.path.join(_ROOT, "models", "i26c_ep31", "t2i26c_ep31.pt"))
    ap.add_argument("--policy", default="broad", choices=("broad", "uniform", "niche"))
    ap.add_argument("--ckpt", default="", help="explicit comma-separated checkpoint paths (overrides --tag)")
    ap.add_argument("--arm_kind", default="concept", choices=("concept", "item"),
                    help="item = matched-budget ITEM interview through the same protocol")
    ap.add_argument("--keep_meh", action="store_true",
                    help="MUST match how the checkpoint was trained (stored in the ckpt as keep_meh)")
    ap.add_argument("--bank", default="random", choices=("random", "pop", "entropy", "polar", "expinfo"),
                    help="RANKED question bank shared by both arms: pop = most answerable, "
                         "entropy = most discriminative. 'random' keeps the old --policy draw "
                         "(concept arm only) and is NOT comparable to the item arm.")
    ap.add_argument("--n_users", type=int, default=1500)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--max_users", type=int, default=0)
    a = ap.parse_args()
    a.lam = 0.0; a.arm = "s1"; a.tau = 1.0; a.eps_l2 = 0.0; a.m_max = 16

    S = CD._build_train_stack(a)
    mc = np.array([len(S["member_sets"][i]) for i in range(len(S["tags"]))], np.float64)
    from train_tower_t2 import compute_head_mask
    import metrics as M
    tr = M.load_train(S["ni"], os.path.join(_ROOT, "data", "ml-25m", "proc"))
    head_mask, _ = compute_head_mask(tr, S["ni"])

    banks = build_banks(S) if a.bank in ("entropy", "polar", "expinfo") else None
    if a.arm_kind == "item":
        if a.bank in ("entropy", "polar"):
            item_rank = banks["item_entropy" if a.bank == "entropy" else "item_polar"]
        else:
            item_rank = np.argsort(-np.asarray(tr.sum(axis=0)).ravel())
        log(f"building COLD ITEM interviews (bank={a.bank}, unrated burns the question) ...")
        packs = cold_item_interviews(S, item_rank, n_users=a.n_users)
        log(f"  {len(packs[0])} users x budgets {MS}")
        r = score(None, S, packs, head_mask)          # items ride the tower; no concept net
        log(f"ITEM ARM bank={a.bank} " + "  ".join(f"m{m}={r[m][0]:.4f}/{r[m][1]:.4f}" for m in MS))
        json.dump({str(m): r[m] for m in MS},
                  open(os.path.join(_ROOT, "experiments", "instrument",
                                    f"item_arm_{a.bank}_coldcheck.json"), "w"), indent=2)
        return

    # EVERY named bank must resolve to a ranking. `expinfo` was missing here and fell through to
    # crank=None, i.e. RANDOM draws, while still logging "bank=expinfo" -- so a run labelled with a
    # ranked bank silently measured a different contract. Assert rather than trust the chain.
    crank = {"entropy": lambda: banks["conc_entropy"],
             "polar":   lambda: banks["conc_polar"],
             "expinfo": lambda: banks["conc_expinfo"],
             "pop":     lambda: np.argsort(-mc),
             "random":  lambda: None}[a.bank]()
    assert (crank is None) == (a.bank == "random"),         f"bank={a.bank} must resolve to a ranking; only 'random' may be unranked"
    log(f"building COLD interviews (zero items, bank={a.bank}, policy={a.policy}) ...")
    packs = cold_interviews(S, mc, n_users=a.n_users, policy=a.policy, rank=crank,
                            keep_meh=a.keep_meh)
    log(f"  {len(packs[0])} users x budgets {MS}")

    # CANARY: with no net at all, every budget must give the SAME number -- the no-evidence prior.
    base = score(None, S, packs, head_mask)
    spread = max(v[0] for v in base.values()) - min(v[0] for v in base.values())
    log(f"CANARY no-net: m0={base[0][0]:.4f}/{base[0][1]:.4f} (full/TAIL) "
        f"spread across budgets={spread:.2e} "
        f"{'PASS' if spread < 1e-9 else 'FAIL -- concepts leaking without a net'}")
    log(f"  BASELINE floor to beat: full={base[0][0]:.4f} tail={base[0][1]:.4f}")

    paths = ([p.strip() for p in a.ckpt.split(",") if p.strip()] if a.ckpt
             else sorted(glob.glob(os.path.join(CKPT_DIR, f"{a.tag}_ep*.pt"))))
    rows = []
    for p in paths:
        blob = torch.load(p, map_location="cpu")
        _h = blob.get("hidden", a.hidden)
        if blob.get("keep_meh", False) != bool(a.keep_meh):
            raise SystemExit(f"MISMATCH: ckpt keep_meh={blob.get('keep_meh', False)} but --keep_meh="
                             f"{bool(a.keep_meh)}. The answer values must match training.")
        net = ConceptFoldNet(len(S["tags"]), d=S["enc"].d_out, h=_h,
                             conc_init=S["d_c"].numpy(), use_refusal=blob.get("use_refusal", False))
        net.load_state_dict(blob["net"])
        r = score(net, S, packs, head_mask)
        sel = float(np.mean([r[m][0] for m in (1, 4, 8)]))
        ok = all(r[m][0] >= r[0][0] for m in (1, 2, 4, 8, 16))       # WITHIN-protocol no-crater
        rows.append({"epoch": blob["epoch"], "sel": sel, "no_crater": ok,
                     **{f"m{m}": r[m] for m in MS}})
        log(f"ep{blob['epoch']:02d} " + "  ".join(f"m{m}={r[m][0]:.4f}/{r[m][1]:.4f}" for m in MS)
            + f" | SEL={sel:.4f} {'' if ok else 'CRATER'}")
    if rows:
        best = max([r for r in rows if r["no_crater"]] or rows, key=lambda r: r["sel"])
        log(f"SELECTED ep{best['epoch']} SEL={best['sel']:.4f} no_crater={best['no_crater']}")
        json.dump(rows, open(os.path.join(_ROOT, "experiments", "instrument",
                                          f"{a.tag}_coldcheck.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
