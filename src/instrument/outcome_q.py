r"""EXPERIMENT Q -- HELD-OUT QUESTIONS: "it knows similarity between concepts and items".

Design sheet: `docs/design/PAPER_B_POLICY_DESIGN.md` **PART H.3 is binding** (it re-instates F.5,
deferred by the author on 2026-08-13 with "train on all and see if it even works" -- it works, so
the deferral expires). Committed before the run per HARD RULE 10.

THE CLAIM UNDER TEST
--------------------
The reaction head predicts a person's reaction to a question it has NEVER SEEN IN TRAINING, from the
question's frozen descriptor alone. A tree cannot represent such a question at all. This is the one
property of the architecture that can be demonstrated without any NDCG comparison.

THE SPLIT (HARD RULE 1: the sampling design is stated in the artifact, not in a comment)
---------------------------------------------------------------------------------------
20% of the 19,390 catalogue questions are held out BY IDENTITY, stratified by CHANNEL (item /
concept) and by POPULARITY DECILE within channel -- 20 strata, 20% held out inside each, so the
held-out set has the same channel mix and the same popularity profile as the training set and the
comparison cannot be won or lost on composition. Popularity = the train rating count for an item,
the train answer rate for a concept. Seed 98765.

The sample-efficiency curve trains on 10 / 25 / 50 / 100% of the REMAINING 80%, drawn with the same
stratification and NESTED (the 10% set is a subset of the 25% set, and so on), so a difference
between two points on the curve is the added questions and nothing else.

★ THE FAIR BASELINE, AND WHY IT NEEDS CARE
------------------------------------------
The per-question popularity prior is the baseline the head had to beat in Part F. **A held-out
question has no per-question row** -- it was never observed in training -- so the prior must BACK
OFF to the mean outcome distribution of its (channel, popularity-decile) stratum, computed over the
TRAINING questions and the FIT users only. That backoff IS the comparator. Comparing against a prior
that has seen the held-out question would be rigged, and the rigged number is reported too, clearly
labelled, only so nobody mistakes the gap for the head's failure.

FALSIFIER, PRE-REGISTERED (unchanged from F.5)
----------------------------------------------
Held-out-question AUC <= the popularity backoff => the similarity claim fails on the easiest
property it will ever be offered, and is REPORTED FAILED. No bigger head, no retry.

Reading the curve: early saturation => "learns from fewer examples via similarity", demonstrated.
Still climbing at 100% => memorisation-shaped, and it is said so.

GUARDS
------
- `--prior_feat` does not exist here. A per-question prior does not exist for a never-seen question,
  so the descriptor is the frozen one and nothing else.
- The no-profile guard (`outcome_head.guard_no_profile`) runs before any gradient step.
- A shuffle control (labels taken from another user) runs at the 100% point.
- FIREWALL: the 10,000 canonical TEST users are never built, never loaded, never named. No NDCG is
  measured anywhere in this file. Every artifact stamps `test_touched=false`.
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

from policy_net import build_phi, N_OUTCOMES, OUTCOME, SEED, log
from outcome_head import (OutcomeModel, load_bank, guard_no_profile, label_block, argmax3, Acc,
                          base_rates, pool_to_catalogue, OUTDIR, TOTAL_L, VAL_M0_FULL, OUT_NAMES)

QDIR = os.path.join(OUTDIR, "heldout_q")


# =============================================================================== the question split
def stratify(ni, C, cnt, ans_rate, n_bucket=10):
    """(channel, popularity-decile) stratum id per catalogue candidate. Items are bucketed by train
    rating count, concepts by train answer rate; deciles are taken WITHIN channel so the two channels
    are never mixed inside a bucket."""
    P = ni + C
    st = np.zeros(P, np.int64)
    for ch, vals, off in ((0, np.asarray(cnt, np.float64), 0), (1, np.asarray(ans_rate, np.float64), ni)):
        n = len(vals)
        order = np.argsort(np.argsort(vals, kind="stable"), kind="stable")
        b = (order * n_bucket // max(n, 1)).clip(0, n_bucket - 1)
        st[off:off + n] = ch * n_bucket + b
    return st


def split_questions(st, frac_heldout=0.20, seed=SEED):
    """Hold out `frac_heldout` of every stratum, by identity."""
    rng = np.random.default_rng(seed)
    held = []
    for s in np.unique(st):
        idx = np.nonzero(st == s)[0]
        rng.shuffle(idx)
        k = int(round(len(idx) * frac_heldout))
        held.append(idx[:k])
    held = np.sort(np.concatenate(held))
    train = np.sort(np.setdiff1d(np.arange(len(st)), held))
    return train, held


def nested_subsets(train_q, st, fracs, seed=SEED):
    """Nested stratified subsets of the training questions: subset(f1) subset(f2) for f1 < f2."""
    rng = np.random.default_rng(seed + 1)
    order = []
    for s in np.unique(st[train_q]):
        idx = train_q[st[train_q] == s]
        idx = idx.copy(); rng.shuffle(idx)
        order.append(idx)
    # interleave the strata so any prefix of `flat` is stratified
    flat = []
    m = max(len(a) for a in order)
    for j in range(m):
        for a in order:
            if j < len(a):
                flat.append(int(a[j]))
    flat = np.asarray(flat, np.int64)
    out = {}
    for f in fracs:
        k = int(round(len(flat) * f))
        out[f] = np.sort(flat[:k])
    return out


# =============================================================================== batching on a subset
def q_block(D, phi_sub, u, qidx, qmap, P_full, ni):
    """All 16 positions of `u`, restricted to the question subset `qidx` (catalogue indices).

    `qmap` maps a catalogue index to its column in the subset, or -1. Asked questions that are not in
    the subset simply do not appear as columns; those that are get excluded from the loss and from
    every metric exactly as in Part F (their outcome is literally in the input).
    """
    nu = len(u)
    T = TOTAL_L - 1
    Z = np.ascontiguousarray(D["Z"][u]).astype(np.float32)
    lab = D["label_idx"][u]; out = D["outcome"][u]
    hq = np.zeros((nu, TOTAL_L, T), np.int64)
    ho = np.zeros((nu, TOTAL_L, T), np.int64)
    hm = np.zeros((nu, TOTAL_L, T), np.float32)
    for j in range(T):
        hq[:, j + 1:, j] = lab[:, j][:, None]
        ho[:, j + 1:, j] = out[:, j][:, None]
        hm[:, j + 1:, j] = 1.0
    pos = np.tile(np.stack([np.arange(TOTAL_L), TOTAL_L - np.arange(TOTAL_L)], 1)
                  .astype(np.float32), (nu, 1, 1))
    y = label_block(D, u, P_full, ni)[:, qidx]                       # (nu, |Q|)
    Q = len(qidx)
    ur, tr_, qr = np.nonzero(hm)
    cc = qmap[hq[ur, tr_, qr]]
    keepm = cc >= 0
    rr = (ur * TOTAL_L + tr_)[keepm]
    flat = np.unique(rr.astype(np.int64) * Q + cc[keepm].astype(np.int64))
    asked_rc = (flat // Q, flat % Q)
    return {"z": torch.from_numpy(Z.reshape(nu * TOTAL_L, -1)),
            "hp": torch.from_numpy(phi_sub[0][hq.reshape(nu * TOTAL_L, T)]),
            "ho": torch.from_numpy(ho.reshape(nu * TOTAL_L, T)),
            "hm": torch.from_numpy(hm.reshape(nu * TOTAL_L, T)),
            "pos": torch.from_numpy(pos.reshape(nu * TOTAL_L, 2)),
            "y": torch.from_numpy(y), "asked_rc": asked_rc, "nu": nu, "P": Q}


def block_ce_q(s, blk, want_acc=False):
    nu, Q, T = blk["nu"], blk["P"], TOTAL_L
    yrep = blk["y"].repeat_interleave(T, 0).long()
    l = torch.logsumexp(s, 0) - s.gather(0, yrep[None]).squeeze(0)
    n = nu * T * Q - len(blk["asked_rc"][0])
    tot = l.sum() - l[blk["asked_rc"]].sum()
    if not want_acc:
        return tot / n, n
    with torch.no_grad():
        hit = (argmax3(s) == yrep)
        acc = float(hit.sum() - hit[blk["asked_rc"]].sum()) / n
    return tot / n, acc


# =============================================================================== evaluation
def evaluate_q(model, D, users, phi_full, qidx, qmap, P_full, ni, priors=None, batch=32,
               log_every=60):
    """Held-out users x 16 positions x the HELD-OUT QUESTIONS. `priors` is a dict name -> (Q,3)
    probability table, each scored through the SAME accumulator, cell for cell."""
    accs = {"head": Acc()}
    for k in (priors or {}):
        accs[k] = Acc()
    prior_t = {k: torch.from_numpy(np.ascontiguousarray(v.T.astype(np.float32)))
               for k, v in (priors or {}).items()}                   # (3,Q)
    phi_sub = np.ascontiguousarray(phi_full[qidx])
    phi_g = torch.from_numpy(phi_sub)
    model.eval()
    t0 = time.time()
    with torch.no_grad():
        kv = model.head.keys(phi_g)
        for st in range(0, len(users), batch):
            u = users[st:st + batch]
            blk = q_block(D, (phi_full, None), u, qidx, qmap, P_full, ni)
            nu = len(u); Q = len(qidx)
            yt = blk["y"].long()
            keep = torch.ones(nu * TOTAL_L, Q, dtype=torch.bool)
            keep[blk["asked_rc"]] = False
            keep = keep.view(nu, TOTAL_L, Q)
            s = model(blk["z"], blk["hp"], blk["ho"], blk["hm"], blk["pos"], kv=kv)
            pr = torch.softmax(s.view(N_OUTCOMES, nu, TOTAL_L, Q), 0)
            for t in range(TOTAL_L):
                accs["head"].add(t, pr[:, :, t], yt, keep[:, t])
                for k, pt in prior_t.items():
                    accs[k].add(t, pt[:, None, :].expand(-1, nu, -1), yt, keep[:, t])
            if (st // batch) % log_every == 0 and st:
                log(f"  [eval] {st:,}/{len(users):,} users ({(time.time() - t0) / 60:.1f}m)")
    return {k: v.report() for k, v in accs.items()}


# =============================================================================== the run
def main():
    ap = argparse.ArgumentParser(description="Experiment Q (Part H.3): held-out questions")
    ap.add_argument("--fracs", default="0.10,0.25,0.50,1.00")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--eval_batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--d_k", type=int, default=64)
    ap.add_argument("--d_h", type=int, default=256)
    ap.add_argument("--n_heldout_users", type=int, default=10000)
    ap.add_argument("--strip_pop", action="store_true",
                    help="ABLATION: zero the two answerability-correlated descriptor columns "
                         "(item log1p(count), concept answer rate) and refit the standardisation. "
                         "Everything else identical. Tests how much of Q's margin is the head "
                         "reading a feature we gave it rather than similarity.")
    ap.add_argument("--tag", default="", help="output stem suffix; two runs never share a path")
    ap.add_argument("--shuffle_at", type=float, default=1.00,
                    help="run the HARD RULE 5 shuffle control at this training fraction")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--snapshot", default=None)
    ap.add_argument("--concept_ckpt", default=None)
    ap.add_argument("--full_threads", action="store_true")
    ap.add_argument("--i_have_author_goahead", action="store_true")
    args = ap.parse_args()
    assert args.i_have_author_goahead, "REFUSING TO RUN without --i_have_author_goahead"
    fracs = [float(x) for x in args.fracs.split(",") if x.strip()]
    if args.threads:
        torch.set_num_threads(int(args.threads))
    torch.manual_seed(SEED); np.random.seed(SEED)
    os.makedirs(QDIR, exist_ok=True)
    t00 = time.time()

    log("#" * 110)
    log("# EXPERIMENT Q -- HELD-OUT QUESTIONS (design sheet Part H.3 / the deferred F.5)")
    log("# FIREWALL: the 10,000 canonical TEST users are NOT built, NOT loaded, NOT touched.")
    log("# NO NDCG IS MEASURED ANYWHERE IN THIS FILE.")
    log("#" * 110)

    D = load_bank()
    guard_no_profile(D)
    N = len(D["uid"])

    from strategy_channel_suite import (build_greedy_env, contract_selftest, genv_mean,
                                        CERT_TOWER, CERT_CONC, CONCSTATS, N_TRAIN_USERS)
    from ndcg_tree import leak_assert
    args.snapshot = args.snapshot or CERT_TOWER
    args.concept_ckpt = args.concept_ckpt or CERT_CONC
    E = build_greedy_env(args.snapshot, "validation", args.concept_ckpt, "outcomeq/desc")
    contract_selftest(E, E["rows"]); leak_assert(E, E["rows"], "outcomeq/desc")
    f0, t0_ = genv_mean(E, [], E["rows"])
    assert round(f0, 8) == VAL_M0_FULL, f"canonical snap FAILED: {f0!r}"
    log(f"[snap] validation m=0 intercept full={f0!r} -- canonical, exact (the env is built ONLY to "
        f"read the frozen descriptors and to snap the ruler)")
    ni, C = E["ctx"].ni, E["C"]
    catalogue = [(0, i) for i in range(ni)] + [(1, c) for c in range(C)]
    P = len(catalogue)
    phi_full, norm = build_phi(E, catalogue)
    strip_info = None
    if args.strip_pop:
        # ★ THE ANSWERABILITY-FEATURE ABLATION (author's objection, 2026-08-15). `build_phi` hands a
        # concept its train ANSWER RATE and an item its log1p(train count) as literal input columns,
        # and under the structural answerability rule popularity is most of what decides whether an
        # item can be answered at all. The "fair" backoff comparator only ever received popularity
        # DECILES, so the head has strictly more of the same information and part of its margin could
        # be reading a feature we handed it rather than similarity. This zeroes exactly those two raw
        # columns and REFITS the standardisation on the zeroed descriptor, changing nothing else.
        d = int(E["ctx"].Wd.shape[1])
        raw = phi_full * norm[1] + norm[0]
        raw[:, d] = 0.0                                  # log1p(train count) -- items
        raw[:, d + 1] = 0.0                              # train answer rate -- concepts
        mu, sd = raw.mean(0, keepdims=True), raw.std(0, keepdims=True) + 1e-6
        phi_full = ((raw - mu) / sd).astype(np.float32)
        norm = (mu, sd)
        strip_info = {"zeroed_columns": [d, d + 1],
                      "what": "log1p(train rating count) for items; train answer rate for concepts",
                      "kept": "the decoder row / cv3 embedding, the concept signed-value std, and "
                              "the channel one-hot",
                      "norm": "refitted on the zeroed descriptor"}
        log(f"[strip_pop] zeroed descriptor columns {d} and {d + 1} "
            f"(item log-popularity, concept answer rate) and refitted the standardisation")
    d_z = E["ctx"].d
    cnt = np.asarray(E["ctx"].cnt, np.float64)
    zc = np.load(CONCSTATS)
    ans_rate = (zc["n"] / N_TRAIN_USERS).astype(np.float64)
    del E
    log(f"[catalogue] {P:,} candidates ({ni:,} items + {C:,} concepts)")

    p2c = pool_to_catalogue(D["pool"], ni)
    D["label_idx"] = p2c[D["label_idx"]].astype(np.int64)

    # ---- the question split ----------------------------------------------------------------------
    st = stratify(ni, C, cnt, ans_rate)
    train_q, held_q = split_questions(st, 0.20, SEED)
    subsets = nested_subsets(train_q, st, fracs, SEED)
    for a, b in zip(sorted(fracs), sorted(fracs)[1:]):
        assert set(subsets[a].tolist()) <= set(subsets[b].tolist()), "the subsets are not nested"
    comp = {"n_catalogue": P, "n_train_questions": int(len(train_q)),
            "n_heldout_questions": int(len(held_q)),
            "heldout_item_share": float((held_q < ni).mean()),
            "train_item_share": float((train_q < ni).mean()),
            "strata": int(len(np.unique(st))),
            "heldout_by_stratum": {int(s): int((st[held_q] == s).sum()) for s in np.unique(st)},
            "subset_sizes": {str(f): int(len(subsets[f])) for f in fracs},
            "design": ("20% of 19,390 questions held out BY IDENTITY, stratified by channel x "
                       "popularity decile (20 strata); nested stratified training subsets")}
    log(f"[split] {json.dumps(comp)[:600]}")

    # ---- held-out USERS --------------------------------------------------------------------------
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(N)
    heldout_u = np.sort(perm[:args.n_heldout_users])
    fit_u = np.sort(perm[args.n_heldout_users:])
    log(f"[users] fit {len(fit_u):,} / held-out {len(heldout_u):,} TRAIN users. Evaluation is "
        f"HELD-OUT QUESTIONS x HELD-OUT USERS -- both axes unseen.")

    # ---- the baselines, computed on the FIT users and the TRAINING questions only -----------------
    tb = time.time()
    B = base_rates(D, fit_u, P, ni)
    counts = B["pop"] * (len(fit_u) + N_OUTCOMES) - 1.0                 # exact per-question counts
    glob = B["global"]
    # the BACKOFF: a held-out question has no row, so it inherits its (channel, popularity-decile)
    # stratum mean, computed over TRAINING questions only. THIS is the fair comparator.
    backoff_tbl = np.zeros((len(np.unique(st)), N_OUTCOMES), np.float64)
    for j, s in enumerate(np.unique(st)):
        m = train_q[st[train_q] == s]
        c = counts[m].sum(0) + 1.0
        backoff_tbl[j] = c / c.sum()
    sidx = {int(s): j for j, s in enumerate(np.unique(st))}
    backoff = np.asarray([backoff_tbl[sidx[int(st[q])]] for q in held_q], np.float64)

    # ---- ★ PRINT THE COMPOSITION BEFORE SPENDING AN HOUR ON IT ----------------------------------
    # A held-out set that is 95% never-answerable would make the AUC comparison meaningless in a way
    # the headline number hides. So: channel split, popularity-decile histogram, and how many of the
    # held-out questions ANY fit user can answer at all.
    ans_cnt = counts[:, OUTCOME["like"]] + counts[:, OUTCOME["dislike"]]
    comp.update({
        "heldout_n_items": int((held_q < ni).sum()),
        "heldout_n_concepts": int((held_q >= ni).sum()),
        "heldout_decile_histogram_items": [int(((st[held_q] == b) & (held_q < ni)).sum())
                                           for b in range(10)],
        "heldout_decile_histogram_concepts": [int((st[held_q] == 10 + b).sum()) for b in range(10)],
        "heldout_questions_answered_by_at_least_one_fit_user":
            int((ans_cnt[held_q] > 0).sum()),
        "heldout_questions_never_answered_by_anyone": int((ans_cnt[held_q] == 0).sum()),
        "train_questions_answered_by_at_least_one_fit_user": int((ans_cnt[train_q] > 0).sum()),
        "heldout_mean_answer_rate": float((ans_cnt[held_q] / max(len(fit_u), 1)).mean()),
        "train_mean_answer_rate": float((ans_cnt[train_q] / max(len(fit_u), 1)).mean()),
        "heldout_answer_rate_quantiles": [float(x) for x in np.percentile(
            ans_cnt[held_q] / max(len(fit_u), 1), [0, 25, 50, 75, 90, 99, 100])],
    })
    log("[composition] " + json.dumps({k: comp[k] for k in (
        "n_catalogue", "n_train_questions", "n_heldout_questions", "heldout_n_items",
        "heldout_n_concepts", "heldout_questions_answered_by_at_least_one_fit_user",
        "heldout_questions_never_answered_by_anyone", "heldout_mean_answer_rate",
        "train_mean_answer_rate", "heldout_answer_rate_quantiles")}, default=float))
    log(f"[composition] held-out item deciles "
        f"{comp['heldout_decile_histogram_items']} | concept deciles "
        f"{comp['heldout_decile_histogram_concepts']}")
    assert comp["heldout_questions_answered_by_at_least_one_fit_user"] > 0, \
        "no held-out question is answerable by anyone -- the comparison would be vacuous"
    # the RIGGED prior: the per-question row of a question the comparator should not have. Reported
    # only so the gap is legible; it is NOT the comparator.
    rigged = (counts[held_q] + 1.0) / (counts[held_q].sum(1, keepdims=True) + N_OUTCOMES)
    globa = np.tile(glob[None, :].astype(np.float64), (len(held_q), 1))
    priors = {"backoff": backoff, "global": globa, "rigged_per_question": rigged}
    log(f"[baseline] backoff over {len(backoff_tbl)} strata; global "
        f"{glob[0]:.5f}/{glob[1]:.5f}/{glob[2]:.5f} ({(time.time() - tb) / 60:.1f}m)")

    qmap_h = np.full(P, -1, np.int64); qmap_h[held_q] = np.arange(len(held_q))

    runs = []
    arms = [(f, "main") for f in fracs]
    if args.shuffle_at:
        arms.append((float(args.shuffle_at), "shuffle"))
    for frac, arm in arms:
        qidx = subsets[frac]
        qmap = np.full(P, -1, np.int64); qmap[qidx] = np.arange(len(qidx))
        stem = f"q{int(round(frac * 100)):03d}_{arm}" + (f"_{args.tag}" if args.tag else "")
        cellp = os.path.join(QDIR, stem + ".json")
        if os.path.exists(cellp):
            b = json.load(open(cellp))
            if b.get("split_composition") == comp:
                log(f"[resume] {stem} already banked -- reusing")
                runs.append(b); continue
            log(f"[resume] {stem} exists under a DIFFERENT split -- refusing to resume")
        log("=" * 100)
        log(f"=== ARM {stem}: training on {len(qidx):,} of {len(train_q):,} training questions "
            f"({frac * 100:.0f}%), {len(held_q):,} held out")
        t0 = time.time()
        model = OutcomeModel(d_z, phi_full.shape[1], state="both", use_pos=True,
                             d_k=args.d_k, d_h=args.d_h)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        phi_sub = np.ascontiguousarray(phi_full[qidx])
        phi_g = torch.from_numpy(phi_sub)
        lab_user = np.arange(N)
        if arm == "shuffle":
            lab_user = np.random.default_rng(SEED + 7).permutation(N)
            log("[arm] SHUFFLE CONTROL: labels come from a DIFFERENT user; every per-question "
                "marginal survives, so the control's ceiling is the backoff prior itself.")
        ub = max(int(args.batch) // TOTAL_L, 1)
        steps = int(np.ceil(len(fit_u) / ub))
        rr = np.random.default_rng(SEED + 3)
        hist = []
        for ep in range(1, args.epochs + 1):
            model.train()
            pe = rr.permutation(len(fit_u))
            tot = seen = hit = 0.0
            te = time.time()
            for si in range(steps):
                u = fit_u[np.sort(pe[si * ub:(si + 1) * ub])]
                if not len(u):
                    continue
                uu = u if arm == "main" else lab_user[u]
                blk = q_block(D, (phi_full, None), u, qidx, qmap, P, ni)
                if arm == "shuffle":
                    blk["y"] = torch.from_numpy(label_block(D, uu, P, ni)[:, qidx])
                s = model(blk["z"], blk["hp"], blk["ho"], blk["hm"], blk["pos"], phi=phi_g)
                loss, acc = block_ce_q(s, blk, want_acc=True)
                opt.zero_grad(); loss.backward(); opt.step()
                ne = len(u) * TOTAL_L
                tot += float(loss) * ne; seen += ne; hit += acc * ne
                if (si + 1) % args.log_every == 0:
                    el = (time.time() - te) / 60
                    log(f"[{stem} ep{ep} {si + 1}/{steps}] loss {tot / seen:.5f} acc {hit / seen:.5f} "
                        f"({el:.1f}m, projected {el / (si + 1) * steps:.1f}m/epoch)")
            hist.append({"epoch": ep, "train_loss": tot / seen, "train_acc": hit / seen,
                         "minutes": round((time.time() - te) / 60, 2)})
            log(f"[{stem} ep{ep}] loss {tot / seen:.5f} acc {hit / seen:.5f} "
                f"({hist[-1]['minutes']:.1f}m)")
        torch.save({"model": model.state_dict(), "frac": frac, "arm": arm, "d_k": args.d_k,
                    "d_h": args.d_h, "d_phi": int(phi_full.shape[1]), "d_z": d_z,
                    "train_questions": qidx, "heldout_questions": held_q, "seed": SEED},
                   os.path.join(QDIR, stem + ".pt"))
        log(f"[{stem}] evaluating on HELD-OUT QUESTIONS x HELD-OUT USERS")
        res = evaluate_q(model, D, heldout_u, phi_full, held_q, qmap_h, P, ni,
                         priors=priors if arm == "main" else {"backoff": backoff},
                         batch=args.eval_batch)
        # ...and, for context only, on the questions this arm DID train on (same held-out users)
        res_seen = evaluate_q(model, D, heldout_u[:2000], phi_full, qidx,
                              qmap, P, ni, priors=None, batch=args.eval_batch)
        blob = {"analysis": "outcome_heldout_questions_H3", "design_sheet":
                "docs/design/PAPER_B_POLICY_DESIGN.md H.3", "test_touched": False,
                "ndcg_measured": False, "prior_feat": False,
                "arm": arm, "train_fraction_of_training_questions": frac,
                "n_train_questions_used": int(len(qidx)),
                "n_heldout_questions": int(len(held_q)),
                "n_fit_users": int(len(fit_u)), "n_heldout_users": int(len(heldout_u)),
                "split_composition": comp, "history": hist,
                "heldout_question_results": res,
                "seen_question_results_2000users": res_seen["head"],
                "beats_backoff": bool(res["head"]["auc_macro"] > res["backoff"]["auc_macro"]),
                "minutes": round((time.time() - t0) / 60, 2)}
        with open(cellp + ".tmp", "w") as fh:
            json.dump(blob, fh, indent=2, default=float)
        os.replace(cellp + ".tmp", cellp)
        runs.append(blob)
        log(f"[{stem}] HELD-OUT-QUESTION AUC macro head {res['head']['auc_macro']:.5f} vs backoff "
            f"{res['backoff']['auc_macro']:.5f} | CE {res['head']['mean_ce']:.5f} vs "
            f"{res['backoff']['mean_ce']:.5f} | ECE {res['head']['ece_macro']:.5f} | "
            f"beats_backoff={blob['beats_backoff']} ({blob['minutes']:.1f}m)")
        out = {"analysis": "outcome_q_H3", "citable": True, "test_touched": False,
               "split_composition": comp, "runs": runs,
               "strip_pop": strip_info,
               "falsifier": ("held-out-question AUC <= the popularity backoff => the similarity "
                             "claim FAILS and is reported failed"),
               "minutes": round((time.time() - t00) / 60, 2)}
        p = os.path.join(QDIR, "heldout_questions" + (f"_{args.tag}" if args.tag else "") + ".json")
        with open(p + ".tmp", "w") as fh:
            json.dump(out, fh, indent=2, default=float)
        os.replace(p + ".tmp", p)
    log(f"[done] {round((time.time() - t00) / 60, 2)} minutes")
    return out


if __name__ == "__main__":
    main()
