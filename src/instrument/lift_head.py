r"""PAPER B / PART F -- THE LIFT HEAD  V(outcome | state, question),  AND THE COMBINED P x V POLICY.

Design sheet: `docs/design/PAPER_B_POLICY_DESIGN.md` **PART F**. F.1's score is the decomposition of the
target quantity itself:

    score(q | state) = SUM_o  P(o | q, state) * V(o | q, state)      o in {like, dislike, no-answer}

`P` is the three-way outcome head (`outcome_head.py`), fitted on the 140,768 TRAIN users at zero NDCG
cost. `V` is this file: fitted on gains that are **already banked**, so it costs no new scoring either.

WHAT THE BANKED GAINS ACTUALLY ARE
----------------------------------
The tree's split criterion is `node_size_weighted_mean_full_ndcg10_of_the_ternary_partition`, and a
level-0 entry in a node's CELF heap is

    gain(q) = SUM_o w_o(q) * NDCG(child_o(q))  -  NDCG(node)

★ and the tree's own `criterion_note` states the identity that matters here: **NDCG@10 is per-user, so
the node-size-weighted mean over the ternary partition is ALGEBRAICALLY the node mean.** Verified on
disk: max |weighted - banked| over the 7 internal nodes is 5.6e-17. So the split scalar is the plain
cohort-mean NDCG@10 gain of asking q to everyone in the node -- internal nodes and leaves hold the
SAME quantity, and the tree is myopic in Golbandi's sense. Its adaptivity comes from the children
diverging, not from the criterion.

That identity does NOT weaken the P x V form: `SUM_o w_o * NDCG_o` is still the correct decomposition
of the target, and it is still what the lift head's three outputs mean. It only means the two sidecar
families differ in COMPLETENESS, not in what they measure -- so the survivor sets are usable as a
declared, bias-stated secondary arm rather than being a category error.

★ THE MISSINGNESS, HANDLED RATHER THAN DISCOVERED LATE
------------------------------------------------------
`policy_labels.banked_vectors` warns that a level-k heap survivor set is enriched for LOW gains,
because CELF pops the highest-gain candidates first. That warning is real -- and it does **not** apply
to the set this head trains on. A level-0 entry at a node's own state was measured **before any pop**,
so an internal node whose sidecar carries 995-999 of the 1,000-candidate pool has a COMPLETE, unbiased
exact vector. Measured over the 37 sidecars on disk:

  * the **7 internal split nodes** carry 996-1000 level-0 values each -- COMPLETE and unbiased;
  * the **15 leaf** sidecars carry 160-841 of ~997 each -- partial, and biased LOW.

Both compositions are measured and printed by `--audit`, and both go into the results doc. The honest
cost is stated up front: **the clean training set is 7 distinct states.** The 22-state arm
(complete + survivor) is run as a declared secondary, because how many states a value head needs is
the open sizing question and 7-vs-22 is the only free evidence on it that exists.

GATES (identical to `policy_labels.py`, and not loosened to make data usable)
-----------------------------------------------------------------------------
G-L0 pool identity | G-L1 exact `base` equality | G-L2 per-candidate byte-identity to 0.0e+00 |
G-L3 ternary partition recompute | G-L4 argmax consistency on complete vectors | G-L5 no test.

PRE-REGISTERED GATE (F.2), fixed before looking: the composed score's Spearman against the measured
gain at **held-out states** must be >= +0.40, printed beside the measured criterion's +0.784 and the
training-loss surrogate's -0.149. Below it, HARD RULE 6 -- report a weak method, do not tune.

FIREWALL: validation only; `--split test` does not exist in this parser; no train user is ever scored
for NDCG.
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
import torch.nn as nn

from policy_net import StateEncoder, build_phi, paired_bootstrap, channel_mix
from outcome_head import (OutcomeHead, OutcomeModel, N_OUTCOMES, OUTCOME, OUT_NAMES, SEED, log,
                          TOTAL_L, VAL_M0_FULL, pool_to_catalogue)

OUTDIR = os.path.join(_ROOT, "experiments", "battery")
TRAJ_DIR = os.path.join(OUTDIR, "policy_traj")
TREE_JSON = os.path.join(OUTDIR, "ndcgtree.json")
TREE16_JSON = os.path.join(OUTDIR, "ndcgtree16.json")
TREE_HEAPS = os.path.join(OUTDIR, "ndcgtree_heaps")
TREE16_HEAPS = os.path.join(OUTDIR, "ndcgtree16_heaps")
STATIC_COMBINED = os.path.join(OUTDIR, "greedy_static_policyB_combined.json")
STATIC_ITEMS = os.path.join(OUTDIR, "greedy_static_policyB_items.json")
# Banked headline numbers, asserted rather than trusted.
STATIC_VAL_FULL = 0.25561016113602253
STATIC_VAL_TAIL = 0.12366492106552922
TREE_VAL_GAIN_FULL = 0.027939556166529655
TREE_VAL_GAIN_TAIL = 0.010548197664320469
# The two published reference correlations F.2 wants the lift head printed beside.
RHO_MEASURED_CRITERION = 0.784
RHO_TRAINING_LOSS_SURROGATE = -0.149
RHO_GATE = 0.40


# ============================================================================ the banked material
def scan_heaps():
    """The composition audit. Every sidecar, its level-0 count, and whether that vector is COMPLETE
    (measured before any CELF pop, therefore unbiased) or a survivor set (biased LOW)."""
    tr = json.load(open(TREE_JSON))
    leaves = {l["nid"] for l in json.load(open(TREE16_JSON))["leaf_table"]}
    rows = []
    for d, is16 in ((TREE_HEAPS, False), (TREE16_HEAPS, True)):
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if not fn.startswith("node_"):
                continue
            b = json.load(open(os.path.join(d, fn)))
            nid = fn[5:-5]
            lev = {}
            for a, q, u in b["heap"]:
                lev.setdefault(int(u), 0)
                lev[int(u)] += 1
            n0 = lev.get(0, 0) + 1                        # + the candidate that was picked at level 0
            npool = int(b["sig"]["n_pool"])
            rows.append({"dir": os.path.basename(d), "nid": nid, "is_leaf": nid in leaves,
                         "n_pool": npool, "n_level0": n0, "n_heap": sum(lev.values()),
                         "complete": bool(n0 >= npool),
                         "scalar": "cohort_mean_full_ndcg10_gain",   # identical for both families
                         "base": float(b["base"]), "seq_len": len(b["seq"])})
    return tr, rows


def partition_identity_check(tr):
    """The identity the whole reading rests on, re-derived from the artifact rather than quoted:
    the node-size-weighted mean of the ternary partition IS the node mean."""
    worst = 0.0
    for nid, r in tr["nodes"].items():
        if not r["split"]:
            continue
        cs, cf = r["child_sizes"], r["child_full@10"]
        w = sum(cs[k] * cf[k] for k in cs) / sum(cs.values())
        worst = max(worst, abs(w - float(r["node_full@10_after_splitter"])))
    log(f"[identity] node-size-weighted ternary mean == node mean on all 7 internal nodes, "
        f"max |diff| = {worst:.1e} -- so the banked scalar is the plain cohort-mean NDCG@10 gain, "
        f"and internal and leaf sidecars hold the SAME quantity")
    return worst


def banked_states(E, tr, t16, pool, verify_n, rng):
    """Every state with banked level-0 gains, split into the COMPLETE set (the 7 internal nodes,
    measured before any CELF pop, unbiased) and the SURVIVOR set (the 15 leaves, biased LOW) --
    gated exactly as `policy_labels.py` gates a banked reuse, with no gate loosened."""
    from strategy_channel_suite import set_active_rows, genv_mean
    from ndcg_tree import split_rows
    nodes = tr["nodes"]
    leaves = {l["nid"]: l for l in t16["leaf_table"]}
    rows_of = {"root": [int(r) for r in E["rows"]]}

    def rows_for(nid):
        if nid in rows_of:
            return rows_of[nid]
        parent, _br = nid.rsplit(".", 1)
        prows = rows_for(parent)
        q = tuple(int(x) for x in nodes[parent]["splitter"])
        kids, ar, _z = split_rows(E, prows, q)
        got = {k: len(v) for k, v in kids.items()}
        assert got == nodes[parent]["child_sizes"], (
            f"G-L3 PARTITION MISMATCH at {parent}: {got} != {nodes[parent]['child_sizes']}")
        assert round(ar, 4) == round(float(nodes[parent]["answer_rate"]), 4), \
            f"G-L3 ANSWER-RATE MISMATCH at {parent}"
        for k, v in kids.items():
            rows_of[f"{parent}.{k}"] = v
        return rows_of[nid]

    def level0(path, nid):
        b = json.load(open(path))
        g0 = {}
        for a, q, u in b["heap"]:
            if int(u) == 0:
                g0[(int(q[0]), int(q[1]))] = -float(a)
        g0[(int(b["seq"][0][0]), int(b["seq"][0][1]))] = float(b["gains"][0])
        return b, g0

    out = []
    for nid, rec in nodes.items():
        is_leaf = not rec["split"]
        p = os.path.join(TREE_HEAPS, f"node_{nid}.json")
        assert os.path.exists(p), f"heap sidecar missing for node {nid}"
        b, g0 = level0(p, nid)
        prefix = [tuple(int(x) for x in q) for q in rec["path"]]
        avail = [q for q in pool if q not in set(prefix)]
        rws = rows_for(nid)
        complete = len(g0) >= len(avail)
        # --- G-L1: exact base equality, no tolerance -------------------------------------------
        set_active_rows(E, rws); E["memo"] = {}
        base_now = genv_mean(E, prefix, rws)[0]
        assert base_now == float(b["base"]), (
            f"G-L1 BASE MISMATCH at {nid}: this environment gives {base_now!r}, the banked heap says "
            f"{b['base']!r}. Different rulers -- REFUSING the reuse.")
        # --- G-L4: a COMPLETE vector's argmax IS the tree's pick. A survivor set cannot be checked
        #     this way and is not asserted to -- claiming otherwise would be a fake gate.
        pick = (int(b["seq"][0][0]), int(b["seq"][0][1]))
        if complete:
            arg = max(g0.items(), key=lambda kv: kv[1])[0]
            assert arg == pick, f"G-L4 ARGMAX MISMATCH at {nid}: banked {arg}, tree picked {pick}"
        out.append({"nid": nid, "depth": int(rec["depth"]), "prefix": prefix, "rows": rws,
                    "base": base_now, "gains": g0, "pick": pick, "complete": complete,
                    "is_leaf": is_leaf, "n_level0": len(g0), "n_available": len(avail),
                    "coverage": len(g0) / len(avail)})
        log(f"[gate {nid}@{rec['depth']}] G-L1 base {base_now!r} EXACT | {len(g0)}/{len(avail)} "
            f"level-0 gains ({'COMPLETE, unbiased' if complete else 'SURVIVOR, biased LOW'})"
            f"{' | G-L4 argmax == the tree pick' if complete else ''} | {len(rws)} users")
    set_active_rows(E, E["rows"])
    # --- G-L2: per-candidate byte-identity to 0.0e+00, on a re-evaluation sample -----------------
    # The criterion IS the cohort mean (the partition identity above), so the check is a plain
    # `genv_mean` at prefix+[q] -- the same code path the heap's own value came from.
    nver = 0
    if verify_n:
        for st in out:
            set_active_rows(E, st["rows"]); E["memo"] = {}
            cands = [q for q in pool if q in st["gains"]]
            for k in rng.choice(len(cands), size=min(verify_n, len(cands)), replace=False):
                q = cands[int(k)]
                v = genv_mean(E, st["prefix"] + [q], st["rows"])[0]
                assert v == st["base"] + st["gains"][q], (
                    f"G-L2 BYTE-IDENTITY FAILED at {st['nid']} candidate {q}: a fresh evaluation "
                    f"gives {v!r}, the banked value reconstructs to {st['base'] + st['gains'][q]!r}")
                nver += 1
            log(f"[G-L2 {st['nid']}] {min(verify_n, len(cands))} candidates re-evaluated from "
                f"scratch, every one EXACT (0.0e+00)")
        set_active_rows(E, E["rows"])
    return out, nver


# ============================================================================ the lift head
class LiftModel(nn.Module):
    """V(o | q, state): three values per candidate, one per outcome. Same state encoder and the same
    frozen per-channel-lifted descriptors as the outcome head, so the two heads share a
    representation and the combined score is one object rather than two glued together."""

    def __init__(self, d_z, d_phi, state="both", use_pos=True, d_k=64, d_h=256):
        super().__init__()
        self.enc = StateEncoder(d_z, d_phi, d_h=d_h, mode=state, use_pos=use_pos)
        self.head = OutcomeHead(self.enc.d_out, d_phi, d_k=d_k, d_h=d_h)

    def forward(self, z, hp, ho, hm, pos, phi=None, kv=None):
        return self.head(self.enc(z, hp, ho, hm, pos), phi=phi, kv=kv)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


class CombinedPolicy(nn.Module):
    """score(q | state) = SUM_o P(o | q, state) * V(o | q, state), with the signature
    `policy_net.rollout` already speaks, so the policy is scored by the EXISTING funnel and no second
    scorer exists anywhere."""

    def __init__(self, outcome, lift):
        super().__init__()
        self.outcome, self.lift = outcome, lift
        self._kv = None

    def forward(self, z, hp, ho, hm, pos, phi, asked_mask=None):
        if self._kv is None or self._kv[0] is not phi:                 # bounded: exactly one entry
            with torch.no_grad():
                self._kv = (phi, self.outcome.head.keys(phi), self.lift.head.keys(phi))
        p = torch.softmax(self.outcome(z, hp, ho, hm, pos, kv=self._kv[1]), 0)
        v = self.lift(z, hp, ho, hm, pos, kv=self._kv[2])
        s = (p * v).sum(0)
        if asked_mask is not None:
            s = s.masked_fill(asked_mask, float("-inf"))
        return s


# ============================================================================ state tensors
def val_state_tensors(V, p2c, rows_pos, depth):
    """The (z, ternary history, position) tensors for a set of VALIDATION users at one tree state.

    The history's descriptor block is NOT materialised here -- `hq` (indices) is kept and the
    (n_users, 15, d_phi) block is gathered per use. At the root that block alone would be 123 MB and
    there are 22 states (HARD RULE 11c: bound every per-state cache)."""
    z = torch.from_numpy(np.ascontiguousarray(V["Z"][rows_pos, depth]).astype(np.float32))
    B, T = len(rows_pos), TOTAL_L - 1
    hq = np.zeros((B, T), np.int64); ho = np.zeros((B, T), np.int64)
    hm = np.zeros((B, T), np.float32)
    for j in range(depth):
        hq[:, j] = p2c[V["label_idx"][rows_pos, j]]
        ho[:, j] = V["outcome"][rows_pos, j]
        hm[:, j] = 1.0
    pos = np.tile(np.asarray([[depth, TOTAL_L - depth]], np.float32), (B, 1))
    return (z, hq, torch.from_numpy(ho), torch.from_numpy(hm), torch.from_numpy(pos))


def spearman(a, b):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    ra = ra - ra.mean(); rb = rb - rb.mean()
    return float((ra * rb).sum() / max(np.sqrt((ra ** 2).sum() * (rb ** 2).sum()), 1e-12))


# ============================================================================ fitting the lift head
def state_pack(st, V, phi_t, p2c, pool_cat, out_model):
    """Everything one training state needs, computed ONCE: the per-user state tensors, the candidate
    descriptor slice, the frozen outcome-head probabilities, and the exact conditioned targets."""
    pos_of = {int(r): i for i, r in enumerate(V["uid"])}
    jj = np.asarray([pos_of[int(r)] for r in st["rows"]], np.int64)
    cand = [q for q in pool_cat["pool"] if q in st["gains"]]
    ci = np.asarray([pool_cat["index"][q] for q in cand], np.int64)
    y = torch.tensor([st["base"] + st["gains"][q] for q in cand], dtype=torch.float32)
    z, hq, ho, hm, ps = val_state_tensors(V, p2c, jj, st["depth"])
    phi_c = torch.from_numpy(np.ascontiguousarray(phi_t[ci]))
    pk = {"nid": st["nid"], "z": z, "hq": hq, "ho": ho, "hm": hm, "ps": ps, "phi": phi_c,
          "phi_t": phi_t, "y": y, "cand": cand, "base": st["base"], "n_users": len(jj),
          "target_gain": np.asarray([st["gains"][q] for q in cand], np.float64)}
    with torch.no_grad():
        pk["P"] = torch.softmax(out_model(z, hist_phi(pk), ho, hm, ps, phi=phi_c), 0)
    return pk


def hist_phi(pk):
    return torch.from_numpy(pk["phi_t"][pk["hq"]])


def composed(model, pk, grad=False):
    """mean_u SUM_o P(o|q,u) * V(o|q,u) -- the cohort-level composed value, per candidate."""
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        Vv = model(pk["z"], hist_phi(pk), pk["ho"], pk["hm"], pk["ps"], phi=pk["phi"])
        return (pk["P"] * Vv).sum(0).mean(0)


def fit_lift(packs, d_z, d_phi, args, tag, seed=SEED):
    r"""Fit V on the training states, with an INNER validation split and best-epoch selection.

    ★ Why this is not "tuning until it passes". The first version took the LAST epoch of a constant-LR
    Adam run, and its training loss OSCILLATED by nearly an order of magnitude at the end -- e.g.
    2.822e-04 -> 1.126e-04 -> 8.251e-04 -> 9.471e-05 -> 4.008e-04 over epochs 60..120. A held-out
    Spearman read off whatever the optimiser happened to be doing at epoch 120 is partly a sample of
    optimiser noise, not a measurement of what the data supports -- and the whole point of this
    experiment is to tell "seven states cannot support a value model" apart from "our fit was
    unstable". Those two have opposite consequences for whether more labels are worth buying.

    So: a fraction of each TRAINING state's candidates is held out as an inner signal (never the
    held-out state, which stays untouched), the LR decays on a cosine, and the returned model is the
    best-inner-MSE epoch, not the last. The gate is then applied to a stable estimate, pass or fail.
    """
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    inner = []
    for pk in packs:
        n = len(pk["cand"])
        p = rng.permutation(n)
        k = max(int(round(args.lift_inner_val * n)), 1)
        inner.append((torch.from_numpy(np.sort(p[k:])), torch.from_numpy(np.sort(p[:k]))))
    m = LiftModel(d_z, d_phi, state=args.state, use_pos=not args.no_pos, d_k=args.d_k, d_h=args.d_h)
    opt = torch.optim.Adam(m.parameters(), lr=args.lift_lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.lift_epochs)
    hist, best = [], {"mse": 1e9, "ep": 0, "sd": None}
    for ep in range(1, args.lift_epochs + 1):
        tr_tot = 0.0
        for pk, (itr, _iva) in zip(packs, inner):
            c = composed(m, pk, grad=True)
            loss = ((c[itr] - pk["y"][itr]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tr_tot += float(loss)
        sched.step()
        with torch.no_grad():
            va = float(np.mean([float(((composed(m, pk)[iva] - pk["y"][iva]) ** 2).mean())
                                for pk, (_itr, iva) in zip(packs, inner)]))
        hist.append({"epoch": ep, "train_mse": tr_tot / len(packs), "inner_val_mse": va})
        if va < best["mse"]:
            best = {"mse": va, "ep": ep,
                    "sd": {k: v.detach().clone() for k, v in m.state_dict().items()}}
        if ep % max(args.lift_epochs // 6, 1) == 0:
            log(f"  [lift {tag} ep{ep}] train {hist[-1]['train_mse']:.3e} inner-val {va:.3e} "
                f"(best ep{best['ep']} {best['mse']:.3e})")
    m.load_state_dict(best["sd"])
    return m, hist, best["ep"]


# ============================================================================ the driver
def run(args):
    assert args.i_have_author_goahead, "REFUSING to run without the author's go-ahead."
    torch.manual_seed(SEED); np.random.seed(SEED)
    rng = np.random.default_rng(SEED)
    if args.threads:
        torch.set_num_threads(int(args.threads))
    t00 = time.time()

    # ---------- 1. the composition audit, before any environment is built ------------------------
    tr, comp = scan_heaps()
    partition_identity_check(tr)
    # count over `ndcgtree_heaps` only -- `ndcgtree16_heaps` re-banks the SAME leaf states with the
    # tail extended to 16, so counting both would double-count 15 states.
    own = [r for r in comp if r["dir"] == "ndcgtree_heaps"]
    ncomplete = [r for r in own if r["complete"]]
    nsurv = [r for r in own if not r["complete"]]
    log(f"[audit] {len(own)} heap sidecars at their own state: {len(ncomplete)} COMPLETE level-0 "
        f"vectors (measured before any CELF pop -> UNBIASED), {len(nsurv)} survivor sets "
        f"(CELF pops the highest gains first, so a survivor set is enriched for LOW gains)")
    log(f"[audit] complete exact gains available to train on: "
        f"{sum(r['n_level0'] for r in ncomplete):,} over {len(ncomplete)} distinct states")
    log(f"[audit] survivor-biased level-0 values (secondary arm only): "
        f"{sum(r['n_level0'] for r in nsurv):,} over {len(nsurv)} states "
        f"(coverage {np.mean([r['n_level0'] / r['n_pool'] for r in nsurv]) * 100:.1f}% of the pool "
        f"on average, min {min(r['n_level0'] / r['n_pool'] for r in nsurv) * 100:.1f}%)")

    # ---------- 2. the environment, the ruler, the pool -------------------------------------------
    from strategy_channel_suite import (build_greedy_env, contract_selftest, genv_mean, genv_eval,
                                        CERT_TOWER, CERT_CONC)
    from ndcg_tree import build_pool, leak_assert, branch_of_selftest
    E = build_greedy_env(args.snapshot, "validation", args.concept_ckpt, "lifthead/val")
    contract_selftest(E, E["rows"]); leak_assert(E, E["rows"], "lifthead/val")
    rows = E["rows"]
    assert len(rows) == 10000
    f0, tl0 = genv_mean(E, [], rows)
    assert round(f0, 8) == VAL_M0_FULL, f"canonical snap FAILED: {f0!r}"
    log(f"[snap] validation m=0 intercept full={f0!r} tail={tl0!r} -- canonical, exact")
    pool, _ip, _cp = build_pool(E, 500, 500)
    assert len(pool) == 1000 == tr["n_pool"]
    branch_of_selftest(E, rows[:2000], [pool[0], pool[500], (1, 114)])
    ni, C = E["ctx"].ni, E["C"]
    catalogue = [(0, i) for i in range(ni)] + [(1, c) for c in range(C)]
    phi_t, norm = build_phi(E, catalogue)
    pool_cat = {"pool": [tuple(q) for q in pool],
                "index": {(0, i): i for i in range(ni)} | {(1, c): ni + c for c in range(C)}}
    d_z = E["ctx"].d

    # ---------- 3. the banked material, gated ------------------------------------------------------
    t16 = json.load(open(TREE16_JSON))
    all_states, nver = banked_states(E, tr, t16, [tuple(q) for q in pool], args.verify_n, rng)
    states = [s for s in all_states if s["complete"]]
    surv = [s for s in all_states if not s["complete"]]
    assert len(states) == 7, f"expected 7 COMPLETE banked states, got {len(states)}"
    log(f"[banked] {len(states)} complete states (primary, unbiased) + {len(surv)} survivor states "
        f"(secondary, biased LOW; mean pool coverage "
        f"{np.mean([s['coverage'] for s in surv]) * 100:.1f}%)")

    # ---------- 4. the frozen outcome head ---------------------------------------------------------
    blob = torch.load(args.outcome_ckpt, map_location="cpu")
    assert not blob.get("prior_feat"), (
        "the outcome head was trained with --prior_feat, so its descriptor is wider than this "
        "environment's; the lift head would be reading a different input space")
    om = OutcomeModel(blob["d_z"], blob["d_phi"], state=blob["state"], use_pos=blob["use_pos"],
                      d_k=blob["d_k"], d_h=blob["d_h"])
    om.load_state_dict(blob["model"]); om.eval()
    for p in om.parameters():
        p.requires_grad_(False)
    assert np.abs(np.asarray(blob["norm_mu"]) - norm[0]).max() < 1e-6 and \
           np.abs(np.asarray(blob["norm_sd"]) - norm[1]).max() < 1e-6, \
        "the outcome head's descriptor standardisation is not this environment's -- the two heads " \
        "would be reading different inputs for the same question"
    log(f"[outcome] loaded {os.path.basename(args.outcome_ckpt)} (epoch {blob['epoch']}, "
        f"held-out CE {blob['heldout_ce']:.5f}), FROZEN")

    Vv = {"uid": np.load(os.path.join(TRAJ_DIR, "val_traj.npz"))["uid"]}
    vt = np.load(os.path.join(TRAJ_DIR, "val_traj.npz"))
    Vv.update({"label_idx": vt["label_idx"], "outcome": vt["outcome"],
               "Z": np.load(os.path.join(TRAJ_DIR, "z_val.npy"), mmap_mode="r")})
    assert list(np.asarray(Vv["uid"], np.int64)) == list(np.asarray(rows, np.int64))
    p2c = pool_to_catalogue([tuple(q) for q in pool], ni)

    packs = {s["nid"]: state_pack(s, Vv, phi_t, p2c, pool_cat, om) for s in all_states}
    for s in all_states:
        pk = packs[s["nid"]]
        log(f"[state {s['nid']}@{s['depth']}] {pk['n_users']} users, {len(pk['cand'])} banked "
            f"gains ({'complete' if s['complete'] else 'survivor'}), base {pk['base']:.8f}, "
            f"gain range [{pk['target_gain'].min():+.5f}, {pk['target_gain'].max():+.5f}]")

    # ---------- 5. LEAVE-ONE-STATE-OUT: the pre-registered rho >= +0.40 gate -----------------------
    # The held-out state is always a COMPLETE one, in both arms, so the evaluation is never on a
    # biased vector and the 7-vs-22 comparison isolates the effect of ADDING states.
    # ★ The held-out state is a COMPLETE one in BOTH arms and the SAME seven in both, so the
    # 7-vs-22 contrast is measured on common held-out states and isolates the effect of ADDING
    # training states -- which is the comparison the sizing decision needs.
    seeds = [SEED + 1000 * i for i in range(args.lift_seeds)]

    def loso_arm(train_pool, name):
        rows_ = []
        for s in states:
            trp = [packs[t["nid"]] for t in train_pool if t["nid"] != s["nid"]]
            pk = packs[s["nid"]]
            per_seed = []
            for sd in seeds:
                m, h, bep = fit_lift(trp, d_z, phi_t.shape[1], args, f"{name}-{s['nid']}-s{sd}",
                                     seed=sd)
                c = composed(m, pk).numpy()
                rk = int((pk["target_gain"] > pk["target_gain"][int(np.argmax(c))]).sum()) + 1
                per_seed.append({"seed": sd, "best_epoch": bep,
                                 "spearman": spearman(c, pk["target_gain"]),
                                 "rank_of_head_argmax": rk,
                                 "gain_of_head_argmax": float(
                                     pk["target_gain"][int(np.argmax(c))]),
                                 "rmse_ndcg": float(np.sqrt(((c - pk["y"].numpy()) ** 2).mean())),
                                 "inner_val_mse": h[bep - 1]["inner_val_mse"]})
            rho = [p["spearman"] for p in per_seed]
            rows_.append({"held_out_state": s["nid"], "depth": s["depth"], "n_users": pk["n_users"],
                          "n_candidates": len(pk["cand"]), "n_train_states": len(trp),
                          "spearman": float(np.mean(rho)), "spearman_min": float(np.min(rho)),
                          "spearman_max": float(np.max(rho)), "spearman_sd": float(np.std(rho)),
                          "n_seeds": len(seeds), "per_seed": per_seed,
                          "true_best_gain": float(pk["target_gain"].max()),
                          "gain_of_head_argmax": float(np.mean(
                              [p["gain_of_head_argmax"] for p in per_seed])),
                          "rank_of_head_argmax": float(np.mean(
                              [p["rank_of_head_argmax"] for p in per_seed])),
                          "rmse_ndcg": float(np.mean([p["rmse_ndcg"] for p in per_seed]))})
            log(f"[LOSO/{name} {s['nid']}] {len(trp)} training states, {len(seeds)} seeds -> "
                f"held-out Spearman {rows_[-1]['spearman']:+.4f} "
                f"[{rows_[-1]['spearman_min']:+.4f}, {rows_[-1]['spearman_max']:+.4f}] | "
                f"mean argmax rank {rows_[-1]['rank_of_head_argmax']:.0f}/{len(pk['cand'])} "
                f"(gain {rows_[-1]['gain_of_head_argmax']:+.5f} vs best "
                f"{rows_[-1]['true_best_gain']:+.5f})")
        return rows_

    loso = loso_arm(states, "complete7")
    loso22 = loso_arm(all_states, "all22") if not args.no_survivor_arm else []
    rho_mean = float(np.mean([r["spearman"] for r in loso]))
    gate_pass = bool(rho_mean >= RHO_GATE)
    log("=" * 100)
    log(f"[F.2 GATE] mean held-out-state Spearman {rho_mean:+.4f} over {len(loso)} folds vs the "
        f"pre-registered bar {RHO_GATE:+.2f} -> {'PASS' if gate_pass else 'FAIL'}. On the same axis: "
        f"the measured criterion {RHO_MEASURED_CRITERION:+.3f}, the training-loss surrogate "
        f"{RHO_TRAINING_LOSS_SURROGATE:+.3f}")
    log("[F.2 GATE] per-state, all seven, each citable: " +
        ", ".join(f"{r['held_out_state']}={r['spearman']:+.3f}" for r in loso))
    if loso22:
        log(f"[sizing] adding the 15 survivor states moves the mean held-out Spearman "
            f"{rho_mean:+.4f} -> {np.mean([r['spearman'] for r in loso22]):+.4f}. That delta is the "
            f"only free evidence on how many states a value head needs.")

    # ---------- 6. the final lift head, fitted on all seven ---------------------------------------
    lift, lh, lbep = fit_lift([packs[s["nid"]] for s in states], d_z, phi_t.shape[1], args, "final")
    torch.save({"model": lift.state_dict(), "state": args.state, "use_pos": not args.no_pos,
                "d_k": args.d_k, "d_h": args.d_h, "d_phi": int(phi_t.shape[1]), "d_z": d_z,
                "norm_mu": norm[0], "norm_sd": norm[1], "seed": SEED,
                "trained_on": [s["nid"] for s in states]},
               os.path.join(OUTDIR, "lift_head.pt"))
    insample = [{"state": s["nid"],
                 "spearman": spearman(composed(lift, packs[s["nid"]]).numpy(),
                                      packs[s["nid"]]["target_gain"])} for s in states]

    out = {"analysis": "lift_head_and_combined_policy", "design_sheet":
           "docs/design/PAPER_B_POLICY_DESIGN.md PART F", "citable": True, "test_touched": False,
           "split": "validation", "banked_composition": comp,
           "n_complete_states": len(ncomplete), "n_survivor_states": len(nsurv),
           "n_complete_exact_gains": int(sum(r["n_level0"] for r in ncomplete)),
           "n_survivor_level0_values_not_used": int(sum(r["n_level0"] for r in nsurv)),
           "gates": {"G-L0_pool_contract": E["pool_contract"],
                     "G-L1_base_equality": "exact ==, all 7 internal nodes",
                     "G-L2_candidates_reverified": nver,
                     "G-L3_partition_recompute": "child_sizes + answer_rate exact",
                     "G-L4_argmax_consistency": "complete vector argmax == the tree's pick, 7/7",
                     "G-L5_no_test": "split=validation only"},
           "outcome_ckpt": os.path.basename(args.outcome_ckpt),
           "partition_identity_max_abs_err": partition_identity_check(tr),
           "n_states_complete": len(states), "n_states_survivor": len(surv),
           "scalar_note": ("the banked scalar is the plain cohort-mean full NDCG@10 gain; the "
                           "node-size-weighted ternary mean is algebraically identical to it, so "
                           "internal-node and leaf sidecars hold the SAME quantity and differ only "
                           "in completeness"),
           "loso": loso, "loso_mean_spearman": rho_mean,
           "loso_all22": loso22,
           "loso_all22_mean_spearman": float(np.mean([r["spearman"] for r in loso22]))
           if loso22 else None,
           "F2_gate_bar": RHO_GATE, "F2_gate_pass": gate_pass,
           "reference_rho_measured_criterion": RHO_MEASURED_CRITERION,
           "reference_rho_training_loss_surrogate": RHO_TRAINING_LOSS_SURROGATE,
           "in_sample_spearman": insample, "lift_fit_history": lh, "lift_best_epoch": lbep,
           "lift_seeds": args.lift_seeds, "lift_inner_val_frac": args.lift_inner_val,
           "lift_lr": args.lift_lr, "lift_epochs": args.lift_epochs,
           "estimator_stability_note": (
               "an earlier constant-LR run took the LAST epoch and its training loss oscillated by "
               "~an order of magnitude across the final 60 epochs, so its held-out Spearman was "
               "partly a sample of optimiser noise. This run uses an inner-validation split of the "
               "TRAINING states' candidates, cosine LR decay, best-inner-epoch selection, and "
               "reports the across-seed range -- so that 'seven states cannot support a value "
               "model' can be distinguished from 'the fit was unstable'."),
           "lift_params": lift.n_params(),
           "constraint_note": ("The clean training set is SEVEN states. A failure here is a "
                               "measurement of that constraint, not a verdict on the design."),
           "minutes": round((time.time() - t00) / 60, 2)}

    # ---------- 7. the combined P x V policy, on the real funnel ----------------------------------
    if not args.no_rollout:
        out["policy"] = evaluate_combined(E, om, lift, phi_t, catalogue, pool, rows, ni, args)
    p = os.path.join(OUTDIR, f"lift_head{args.label}.json")
    with open(p + ".tmp", "w") as fh:
        json.dump(out, fh, indent=2, default=float)
    os.replace(p + ".tmp", p)
    log(f"[out] {p} ({out['minutes']:.1f}m)")
    return out


# ============================================================================ the combined policy
def evaluate_combined(E, om, lift, phi_t, catalogue, pool, rows, ni, args):
    """The combined score's OWN interviews on the 10,000 validation users, scored by the existing
    funnel. Comparators in Part F.4's order: bank-alone FIRST (it is the real bar), then the tree,
    then the E1 imitation policy."""
    from policy_net import rollout
    from strategy_channel_suite import genv_eval
    pol = CombinedPolicy(om, lift)
    pool_t = [tuple(q) for q in pool]
    idx = {q: i for i, q in enumerate(catalogue)}
    phi_pool = np.ascontiguousarray(phi_t[[idx[q] for q in pool_t]])

    sc = json.load(open(STATIC_COMBINED))
    sf, stl = genv_eval(E, [(int(a), int(b)) for a, b in sc["sequence"]], rows)
    assert abs(float(np.nanmean(sf)) - STATIC_VAL_FULL) < 1e-12, (
        f"BANKED-HEADLINE ASSERT FAILED: greedy_static_policyB_combined rescores to "
        f"{float(np.nanmean(sf))!r}, banked {STATIC_VAL_FULL}")
    log(f"[banked] greedy_static_policyB_combined reproduces EXACTLY: "
        f"full={float(np.nanmean(sf))!r} tail={float(np.nanmean(stl))!r}")
    si = json.load(open(STATIC_ITEMS))
    if_, itl = genv_eval(E, [(int(a), int(b)) for a, b in si["sequence"]], rows)
    bank_full = paired_bootstrap(sf - if_); bank_tail = paired_bootstrap(stl - itl)
    log(f"[F.4 BAR] bank-alone on validation (combined static - items static): "
        f"full {bank_full['mean']:+.5f} [{bank_full['lo']:+.5f},{bank_full['hi']:+.5f}] | "
        f"tail {bank_tail['mean']:+.5f} [{bank_tail['lo']:+.5f},{bank_tail['hi']:+.5f}]")

    res = {"static_combined": {"full": float(np.nanmean(sf)), "tail": float(np.nanmean(stl))},
           "static_items": {"full": float(np.nanmean(if_)), "tail": float(np.nanmean(itl))},
           "bank_alone_gain_full": bank_full, "bank_alone_gain_tail": bank_tail,
           "tree_val_gain_full": TREE_VAL_GAIN_FULL, "tree_val_gain_tail": TREE_VAL_GAIN_TAIL}

    for name, pl, ph in (("pool1000", pool_t, phi_pool), ("catalogue19390", catalogue, phi_t)):
        t0 = time.time()
        ro = rollout(E, pol, ph, pl, rows)
        f, t = ro["curves"][16]
        trained = set(pool_t)
        oop = np.asarray([[0 if pl[int(j)] in trained else 1 for j in r] for r in ro["asked"]])
        ans = (ro["outs"] != OUTCOME["noanswer"])
        res[name] = {
            "full": float(np.nanmean(f)), "tail": float(np.nanmean(t)),
            "gain_vs_static_combined_full": paired_bootstrap(f - sf),
            "gain_vs_static_combined_tail": paired_bootstrap(t - stl),
            "n_distinct_interviews": len({tuple(r) for r in ro["asked"]}),
            "channel_mix_concept_share": channel_mix(ro["asked"], pl),
            "answer_rate": float(ans.mean()),
            "out_of_pool_pick_rate": float(oop.mean()),
            "out_of_pool_by_position": [float(x) for x in oop.mean(0)],
            "users_with_any_out_of_pool_pick": float((oop.sum(1) > 0).mean()),
            "answer_rate_in_pool_picks": float(ans[~oop.astype(bool)].mean()) if (oop == 0).any() else None,
            "answer_rate_out_of_pool_picks": float(ans[oop.astype(bool)].mean()) if oop.any() else None,
            "minutes": round((time.time() - t0) / 60, 2)}
        g = res[name]["gain_vs_static_combined_full"]
        log(f"[{name}] full {res[name]['full']:.5f} | gain vs combined static {g['mean']:+.5f} "
            f"[{g['lo']:+.5f},{g['hi']:+.5f}] | answer rate {res[name]['answer_rate']:.4f} | "
            f"{res[name]['n_distinct_interviews']} distinct interviews | "
            f"{res[name]['out_of_pool_pick_rate'] * 100:.1f}% out-of-pool "
            f"({res[name]['minutes']:.1f}m)")
    # the open-vocabulary deficit E1 paid: -0.02034 [-0.02236,-0.01837]
    fp = res["catalogue19390"]["gain_vs_static_combined_full"]["mean"]
    pp = res["pool1000"]["gain_vs_static_combined_full"]["mean"]
    res["open_vocab_deficit_full"] = fp - pp
    res["E1_open_vocab_deficit_full"] = -0.02034
    res["beats_bank_alone_full"] = bool(
        res["pool1000"]["gain_vs_static_combined_full"]["mean"] > bank_full["mean"])
    res["beats_tree_full"] = bool(
        res["pool1000"]["gain_vs_static_combined_full"]["mean"] > TREE_VAL_GAIN_FULL)
    return res


def main():
    ap = argparse.ArgumentParser(description="Part F lift head + combined P x V policy")
    ap.add_argument("--audit", action="store_true",
                    help="print the banked-heap composition and exit. No environment, no fit.")
    ap.add_argument("--outcome_ckpt", default=os.path.join(OUTDIR, "outcome_head_main_both.pt"))
    ap.add_argument("--state", default="both", choices=["z", "hist", "both"])
    ap.add_argument("--no_pos", action="store_true")
    ap.add_argument("--d_k", type=int, default=64)
    ap.add_argument("--d_h", type=int, default=256)
    ap.add_argument("--lift_epochs", type=int, default=80)
    ap.add_argument("--lift_lr", type=float, default=1e-4)
    ap.add_argument("--lift_seeds", type=int, default=3,
                    help="fits per fold; the Spearman is reported as a range across them")
    ap.add_argument("--lift_inner_val", type=float, default=0.2,
                    help="fraction of each TRAINING state's candidates held out for best-epoch "
                         "selection. Never the held-out state, which stays untouched.")
    ap.add_argument("--label", default="", help="suffix for the output artifact")
    ap.add_argument("--verify_n", type=int, default=3,
                    help="G-L2: candidates per state re-evaluated from scratch to 0.0e+00")
    ap.add_argument("--no_rollout", action="store_true")
    ap.add_argument("--no_survivor_arm", action="store_true",
                    help="skip the 22-state secondary arm (complete + survivor-biased)")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--snapshot", default=None)
    ap.add_argument("--concept_ckpt", default=None)
    ap.add_argument("--full_threads", action="store_true")
    ap.add_argument("--i_have_author_goahead", action="store_true")
    args = ap.parse_args()
    if args.audit:
        tr, comp = scan_heaps()
        for r in comp:
            log(f"  {r['dir']:18s} {r['nid']:46s} lev0 {r['n_level0']:4d}/{r['n_pool']} "
                f"{'COMPLETE' if r['complete'] else 'survivor'} {r['scalar']}")
        return comp
    from strategy_channel_suite import CERT_TOWER, CERT_CONC
    args.snapshot = args.snapshot or CERT_TOWER
    args.concept_ckpt = args.concept_ckpt or CERT_CONC
    log("#" * 110)
    log("# PART F -- THE LIFT HEAD V(o | state, question) AND THE COMBINED P x V POLICY")
    log("# Fitted on ALREADY-BANKED exact conditioned gains. NO new scoring. TEST never built.")
    log("#" * 110)
    return run(args)


if __name__ == "__main__":
    main()
