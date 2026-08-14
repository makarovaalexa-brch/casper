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

★ THE LEVEL-0 EXTRACTION TRAP, because it silently deletes the best candidate at every state:
filtering heap entries on `upd == 0` alone does **NOT** give the complete level-0 vector. CELF POPS the
candidate it selects, so the selected one is no longer in the heap and must be added back from
`seq[0]` / `gains[0]`. Miss it and every training target is missing the single HIGHEST-gain candidate
at every state, and the root argmax comes out as [1,994] instead of [1,114]. `level0()` adds it back,
and **G-L4 is the gate that catches the mistake**: the complete vector's argmax must equal the tree's
recorded splitter, which fails immediately if the pick was dropped. Verified here: 1000/1000 at the
root, argmax [1,114] at gain 0.02049056, byte-matching `ndcgtree.json`.

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


def value(model, pk, offset=True):
    r"""V(o | q, u) for every user and candidate of a state, as a (3, n_users, n_cand) tensor.

    ★ `offset=True` is the structural reparameterisation, and it is a reduction in what SEVEN states
    have to teach rather than a tuning knob. The head emits `delta_o(q, s)` and

        V(o | q, s) = base(s) + delta_o(q, s)

    so the composed value is `base(s) + SUM_o P(o) * delta_o` and the quantity being FIT is the GAIN
    (order 1e-2) instead of the level (0.05..0.40). The level is a known constant per state -- it is
    the heap's own `base`, gated to exact equality by G-L1 -- so making the head relearn it from seven
    examples was spending the entire label budget on something already on disk.

    It also puts the environment's own near-analytic constraint at the origin: a question that is NOT
    answered folds an UNSEEN/refusal token, so the state barely moves and `V(no-answer|q,s)` should sit
    close to `base(s)`, i.e. `delta_noanswer ~ 0`, near-flat across q. Under `offset=True` that is the
    zero-initialised default; under free-form prediction it is something the head must discover.
    """
    d = model(pk["z"], hist_phi(pk), pk["ho"], pk["hm"], pk["ps"], phi=pk["phi"])
    return d + pk["base"] if offset else d


def composed(model, pk, grad=False, offset=True):
    """mean_u SUM_o P(o|q,u) * V(o|q,u) -- the cohort-level composed value, per candidate."""
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        return (pk["P"] * value(model, pk, offset)).sum(0).mean(0)


def diagnose_value_head(model, packs, states, offset=True, topk=10):
    r"""★ THE STRUCTURAL CORRECTNESS TEST, run before any Spearman is believed.

    The environment gives a near-analytic constraint the head must satisfy for its output to mean
    anything: an unanswered question folds an UNSEEN/refusal token, so the state hardly moves and
    `V(no-answer | q, s)` should sit close to `base(s)` and be near-FLAT across q. That is a property
    of the environment, not of the fit.

    It matters arithmetically, not just aesthetically: **P(no-answer) is 0.966 over the catalogue**, so
    in `SUM_o P(o) V(o)` the no-answer branch dominates the product by construction. If V(no-answer) is
    over-estimated -- or merely VARIES across q when it should be flat -- the score is maximised by
    questions that will NOT be answered. That is the exact signature measured in the first run: answer
    rate 0.1241 in-pool, 0.0092 open-vocab, concept share 0.0000, 100% out-of-pool picks.

    Reports, per state: the mean and spread of each outcome branch against `base(s)`, and a
    decomposition of the top-k combined-score picks into their P and V factors, so it is visible
    whether the argmax is driven by the probability term or the value term.
    """
    out = []
    for st in states:
        pk = packs[st["nid"]]
        with torch.no_grad():
            V = value(model, pk, offset)                     # (3, n_users, n_cand)
            Pm = pk["P"].mean(1)                             # (3, n_cand) cohort-mean probability
            Vm = V.mean(1)                                   # (3, n_cand) cohort-mean value
            comp = (pk["P"] * V).sum(0).mean(0).numpy()
        rec = {"state": st["nid"], "base": float(pk["base"]),
               "banked_child_full": st.get("child_full"),
               "branches": {}}
        for o, nm in enumerate(OUT_NAMES):
            v = Vm[o].numpy()
            rec["branches"][nm] = {
                "mean": float(v.mean()), "sd_across_q": float(v.std()),
                "min": float(v.min()), "max": float(v.max()),
                "mean_minus_base": float(v.mean() - pk["base"]),
                "mean_P": float(Pm[o].mean())}
        # is the argmax driven by P or by V?
        order = np.argsort(-comp)[:topk]
        rec["top_picks"] = [{"q": list(pk["cand"][int(j)]),
                             "composed": float(comp[int(j)]),
                             "true_gain": float(pk["target_gain"][int(j)]),
                             "P": [float(Pm[o, int(j)]) for o in range(N_OUTCOMES)],
                             "V": [float(Vm[o, int(j)]) for o in range(N_OUTCOMES)]}
                            for j in order]
        # correlation of the composed score with each factor, to name the driver
        rec["corr_composed_with"] = {
            f"P_{OUT_NAMES[o]}": spearman(comp, Pm[o].numpy()) for o in range(N_OUTCOMES)}
        rec["corr_composed_with"].update({
            f"V_{OUT_NAMES[o]}": spearman(comp, Vm[o].numpy()) for o in range(N_OUTCOMES)})
        out.append(rec)
        na = rec["branches"]["noanswer"]
        log(f"[Vdiag {st['nid']}] base {pk['base']:.5f} | no-answer branch mean "
            f"{na['mean']:.5f} (base{na['mean_minus_base']:+.5f}) sd across q {na['sd_across_q']:.5f} "
            f"| like {rec['branches']['like']['mean']:.5f} dislike "
            f"{rec['branches']['dislike']['mean']:.5f} | mean P(no-answer) {na['mean_P']:.4f}")
    return out


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
            c = composed(m, pk, grad=True, offset=args.v_offset)
            loss = ((c[itr] - pk["y"][itr]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tr_tot += float(loss)
        sched.step()
        with torch.no_grad():
            va = float(np.mean([float(((composed(m, pk, offset=args.v_offset)[iva]
                                          - pk["y"][iva]) ** 2).mean())
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


# ================================================= HARD RULE 2: the simplest possible estimator
def mean_vector_loso(states, packs, label="complete7"):
    r"""★ THE BASELINE THAT SHOULD HAVE BEEN RUN FIRST (HARD RULE 2, and it was missed).

    Predict a held-out state's whole gain vector with the **plain mean of the other states' gain
    vectors**. No features, no parameters, no learning, and -- the part that matters -- **no state
    information at all**: it does not know which state it is predicting for.

    If this clears the gate that the learned head fails, then "the descriptor and the interview state
    cannot resolve measured NDCG gain" is false, and so is "seven states are not enough". It is free,
    it takes milliseconds, and it should have preceded every fit in this file.
    """
    rows = []
    for s in states:
        pk = packs[s["nid"]]
        others = [packs[t["nid"]] for t in states if t["nid"] != s["nid"]]
        pred, keep = [], []
        for j, q in enumerate(pk["cand"]):
            vals = [float(o["target_gain"][o["cand"].index(q)]) for o in others if q in o["cand"]]
            if vals:
                pred.append(float(np.mean(vals))); keep.append(j)
        y = pk["target_gain"][np.asarray(keep, np.int64)]
        pred = np.asarray(pred)
        rk = int((y > y[int(np.argmax(pred))]).sum()) + 1
        rows.append({"held_out_state": s["nid"], "depth": s["depth"],
                     "n_candidates_scored": len(keep), "n_train_states": len(others),
                     "spearman": spearman(pred, y), "rank_of_argmax": rk,
                     "true_best_gain": float(y.max()),
                     "gain_of_argmax": float(y[int(np.argmax(pred))])})
        log(f"[MEANVEC/{label} {s['nid']}] {len(others)} states averaged -> held-out Spearman "
            f"{rows[-1]['spearman']:+.4f} | argmax rank {rk}/{len(keep)} "
            f"(gain {rows[-1]['gain_of_argmax']:+.5f} vs best {rows[-1]['true_best_gain']:+.5f})")
    return rows


def mean_gain_table(states, packs, pool):
    """The static per-question value table: the mean measured gain of each pool candidate over the
    states that hold a value for it. Derived from the VALIDATION cohort, exactly like the greedy
    static order it is compared against."""
    acc = {}
    for s in states:
        pk = packs[s["nid"]]
        for j, q in enumerate(pk["cand"]):
            acc.setdefault(q, []).append(float(pk["target_gain"][j]))
    g = np.zeros(len(pool), np.float32)
    n = np.zeros(len(pool), np.int64)
    for i, q in enumerate(pool):
        if q in acc:
            g[i] = float(np.mean(acc[q])); n[i] = len(acc[q])
    return g, n


class MeanVectorPolicy(nn.Module):
    r"""The two-head policy with the LEARNED value half replaced by the parameter-free mean vector,
    and the working outcome head left untouched.

    Same `V = base(s) + delta` form. The decomposition is pinned by the environment rather than fitted:
    `delta_noanswer = 0` (an unanswered question folds an UNSEEN/refusal token, so the state barely
    moves) and the remaining value is carried equally by the two ANSWERED branches, so

        gain_cohort(q) = abar(q) * delta(q)   =>   delta(q) = gain_cohort(q) / abar(q)

    and the per-user score is

        score(q | s, u) = base(s) + gain(q) * P(answered | q, s, u) / abar(q)

    i.e. the static measured value of a question, reweighted by how much MORE likely this user is to
    answer it than the cohort average. That is the two-head design doing the job it was proposed for --
    pricing answerability per user -- with an estimator of the value half that is not broken.

    `--value meanvec_static` drops the outcome head entirely and asks the argmax of `gain(q)`, which is
    a STATIC order. The gap between the two arms is exactly what the outcome head contributes.
    """

    def __init__(self, outcome, gain, abar, use_outcome=True):
        super().__init__()
        self.outcome = outcome
        self.register_buffer("gain", torch.from_numpy(np.asarray(gain, np.float32)))
        self.register_buffer("delta", torch.from_numpy(
            (np.asarray(gain, np.float32) / np.maximum(np.asarray(abar, np.float32), 1e-3))))
        self.use_outcome = use_outcome

    def forward(self, z, hp, ho, hm, pos, phi, asked_mask=None):
        if self.use_outcome:
            p = torch.softmax(self.outcome(z, hp, ho, hm, pos, phi=phi), 0)
            s = (p[0] + p[1]) * self.delta[None, :]
        else:
            s = self.gain[None, :].expand(z.shape[0], -1).clone()
        if asked_mask is not None:
            s = s.masked_fill(asked_mask, float("-inf"))
        return s


# ============================ THE CEILING EXPERIMENT: policy score vs value-estimator accuracy
def noisy_value_table(t, rho_target, seed, tol=0.004, iters=40):
    r"""A value vector with the SAME marginal distribution of values as `t` but a controlled Spearman
    correlation to it.

    Rank-preserving by construction: the perturbation is applied in normal-score space and the result
    is mapped back onto `sorted(t)`, so the estimator being simulated is wrong about the ORDER of
    questions by exactly the target amount while remaining a plausible value vector -- it cannot cheat
    by shrinking toward a constant, which would flatten the policy for reasons that have nothing to do
    with accuracy.

    The Gaussian latent correlation is solved by bisection against the realised Spearman rather than
    assumed from the (slightly wrong) `rho = (6/pi) arcsin(rho_latent/2)` identity.
    """
    rng = np.random.default_rng(seed)
    n = len(t)
    order = np.argsort(np.argsort(t))
    z = (order + 0.5) / n
    z = np.sqrt(2.0) * _erfinv(2.0 * z - 1.0)                   # normal scores of t's ranks
    e = rng.standard_normal(n)
    srt = np.sort(t)
    lo, hi = 0.0, 1.0
    best = None
    for _ in range(iters):
        a = 0.5 * (lo + hi)
        zz = a * z + np.sqrt(max(1.0 - a * a, 0.0)) * e
        v = srt[np.argsort(np.argsort(zz))]
        r = spearman(v, t)
        best = (v, r, a)
        if abs(r - rho_target) <= tol:
            break
        if r < rho_target:
            lo = a
        else:
            hi = a
    return best[0].astype(np.float32), float(best[1])


def _erfinv(x):
    """Winitzki's approximation, refined by two Newton steps -- avoids a scipy dependency."""
    a = 0.147
    ln = np.log(np.clip(1.0 - x * x, 1e-300, None))
    t1 = 2.0 / (np.pi * a) + ln / 2.0
    y = np.sign(x) * np.sqrt(np.maximum(np.sqrt(t1 * t1 - ln / a) - t1, 0.0))
    for _ in range(2):
        err = _erf(y) - x
        y = y - err / (2.0 / np.sqrt(np.pi) * np.exp(-y * y))
    return y


def _erf(x):
    """Abramowitz & Stegun 7.1.26."""
    sgn = np.sign(x); x = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * x)
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
               + 0.254829592) * t * np.exp(-x * x)
    return sgn * y


# ================== ALL BANKED STATES: the level>0 material the gate test deliberately excluded
SANITY_MARKERS = ("SANITY", "PREFLIGHT")


def enumerate_all_banked_states():
    r"""Every state with banked gains, across every heap sidecar on disk, de-duplicated properly.

    ★ A STATE IS (cohort, asked-question-set) -- NOT (sidecar, level). Keying on the latter
    double-counts, because `ndcgtree16_heaps` re-banks the same leaf states with the tail extended to
    16: the same prefix, the same early picks, the same cohort. Independently counted here, the
    difference is large: **585 (sidecar, level) pairs and ~53.9k raw heap entries collapse to 253
    distinct states and 24,013 measured (state, question, gain) triples.**

    Where two sidecars measured the SAME state, their gains are MERGED and any question they share is
    asserted EQUAL -- a free cross-run byte-identity check that costs nothing and would catch a
    contract drift between two independently-launched jobs.

    FIREWALL / hygiene, both enforced rather than assumed:
      * every sidecar must carry `sig["split"] == "validation"` and a `:validation` contract tag.
        (`ndcgtree16test_heaps` is validation material despite its name -- same `rows_sha1`, same base,
        same prefix as the validation bank; the "test" refers to the driver job, not the cohort. It is
        admitted on the contract, not on the filename.)
      * `ndcgtree_SANITY_heaps` / `_PREFLIGHT_heaps` are EXCLUDED: the SANITY tree is explicitly not
        the citable tree (`policy_traj.load_tree` refuses it), so its states are not citable material.

    ⚠ THE BIAS, stated once and carried: level-0 vectors at a node's own state are complete and
    unbiased, but every level>0 entry is a CELF SURVIVOR -- the high-gain candidates were popped
    first, so a level-k set is enriched for LOW gains. That is truncation on the dependent variable.
    It is the reason these states were excluded from the GATE test, and it is why the evaluation here
    is still only ever on the 7 COMPLETE states.
    """
    import glob
    out = {}
    n_files = n_raw = 0
    for d in sorted(glob.glob(os.path.join(OUTDIR, "*_heaps"))):
        base_d = os.path.basename(d)
        if any(m in base_d for m in SANITY_MARKERS):
            log(f"[allstates] EXCLUDING {base_d} (not the citable tree)")
            continue
        for fn in sorted(os.listdir(d)):
            if not fn.endswith(".json"):
                continue
            b = json.load(open(os.path.join(d, fn)))
            if "heap" not in b or "sig" not in b:
                continue
            sig = b["sig"]
            # FIREWALL. Two sidecar signature versions exist: v1 carries `contract_tag`, v2 carries
            # `mask_pool`. Both carry `split`. Require `split == validation`, and where a contract tag
            # exists require it to name validation too -- admitted on the CONTRACT, never the filename.
            assert sig.get("split") == "validation",                 f"FIREWALL: {base_d}/{fn} is not validation material (split={sig.get('split')!r})"
            ct = sig.get("contract_tag")
            assert ct is None or ":validation" in str(ct),                 f"FIREWALL: {base_d}/{fn} contract tag {ct!r} does not name validation"
            mp = sig.get("mask_pool")
            assert mp is None or mp.get("policy") == "B",                 f"{base_d}/{fn} is not a policy-B mask ({mp})"
            n_files += 1
            prefix = [tuple(int(x) for x in q) for q in sig.get("prefix", [])]
            seq = [tuple(int(x) for x in q) for q in b["seq"]]
            rows_key = f"{sig.get('rows_sha1')}:{sig.get('n_rows')}"
            lev = {}
            for a, q, u in b["heap"]:
                lev.setdefault(int(u), {})[(int(q[0]), int(q[1]))] = -float(a)
            cur = float(b["base"])
            for k in range(len(seq)):
                g = dict(lev.get(k, {}))
                g[seq[k]] = float(b["gains"][k])                 # the popped pick, added back
                n_raw += len(g)
                key = (rows_key, tuple(sorted(prefix + seq[:k])))
                rec = out.get(key)
                if rec is None:
                    out[key] = {"asked": prefix + seq[:k], "base": cur, "gains": g,
                                "n_rows": int(sig["n_rows"]), "rows_sha1": sig.get("rows_sha1"),
                                "src": [f"{base_d}/{fn}@{k}"], "node": fn[5:-5] if
                                fn.startswith("node_") else None, "level": k}
                else:
                    # the same state measured twice: bases must agree, shared gains must agree
                    assert abs(rec["base"] - cur) < 1e-12, (
                        f"CROSS-RUN BASE MISMATCH at {key[1][:2]}...: {rec['base']!r} vs {cur!r} "
                        f"({rec['src'][0]} vs {base_d}/{fn}@{k})")
                    for q, v in g.items():
                        if q in rec["gains"]:
                            assert abs(rec["gains"][q] - v) < 1e-12, (
                                f"CROSS-RUN GAIN MISMATCH at {q}: {rec['gains'][q]!r} vs {v!r}")
                        else:
                            rec["gains"][q] = v
                    rec["src"].append(f"{base_d}/{fn}@{k}")
                cur += float(b["gains"][k])
    n_tri = sum(len(v["gains"]) for v in out.values())
    log(f"[allstates] {n_files} sidecars -> {len(out)} DISTINCT states "
        f"(cohort + asked-set), {n_tri:,} measured (state, question, gain) triples")
    log(f"[allstates] raw heap entries before de-duplication: {n_raw:,} -- the gap is the "
        f"(sidecar, level) double count the docstring warns about")
    sz = sorted(len(v["gains"]) for v in out.values())
    log(f"[allstates] candidates per state: min {sz[0]} p25 {sz[len(sz)//4]} "
        f"median {sz[len(sz)//2]} p75 {sz[3*len(sz)//4]} max {sz[-1]} -- most states carry very few, "
        f"and those few are the CELF survivors, i.e. the LOW-gain tail")
    return out


def build_state_pack_live(E, st, phi_t, pool_cat, om, ni, rows):
    """A training pack for an arbitrary banked state: fold its asked list live, take each user's own
    ternary outcomes for those questions, and score the frozen outcome head over the candidates that
    state holds gains for."""
    from strategy_channel_suite import set_active_rows, genv_latent
    from ndcg_tree import branch_of_per_user
    cand = [q for q in st["gains"] if q in pool_cat["index"]]
    if not cand:
        return None
    ci = np.asarray([pool_cat["index"][q] for q in cand], np.int64)
    asked = [tuple(q) for q in st["asked"]]
    set_active_rows(E, rows)
    z = genv_latent(E, asked, rows)
    n, T = len(rows), TOTAL_L - 1
    hq = np.zeros((n, T), np.int64); ho = np.zeros((n, T), np.int64)
    hm = np.zeros((n, T), np.float32)
    for j, q in enumerate(asked[:T]):
        hq[:, j] = pool_cat["index"][tuple(q)]
        ho[:, j] = branch_of_per_user(E, rows, [tuple(q)] * n)
        hm[:, j] = 1.0
    t = min(len(asked), TOTAL_L)
    pos = np.tile(np.asarray([[t, TOTAL_L - t]], np.float32), (n, 1))
    phi_c = torch.from_numpy(np.ascontiguousarray(phi_t[ci]))
    pk = {"nid": st.get("node") or f"state@{t}", "z": z, "hq": hq,
          "ho": torch.from_numpy(ho), "hm": torch.from_numpy(hm),
          "ps": torch.from_numpy(pos), "phi": phi_c, "phi_t": phi_t,
          "y": torch.tensor([st["base"] + st["gains"][q] for q in cand], dtype=torch.float32),
          "cand": cand, "base": float(st["base"]), "n_users": n,
          "target_gain": np.asarray([st["gains"][q] for q in cand], np.float64)}
    with torch.no_grad():
        pk["P"] = torch.softmax(om(z, hist_phi(pk), pk["ho"], pk["hm"], pk["ps"], phi=phi_c), 0)
    return pk


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
                c = composed(m, pk, offset=args.v_offset).numpy()
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
                 "spearman": spearman(composed(lift, packs[s["nid"]],
                                               offset=args.v_offset).numpy(),
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
           "v_offset": bool(args.v_offset),
           "value_head_diagnostics": diagnose_value_head(
               lift, packs, states, offset=args.v_offset),
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


def run_meanvec(args):
    """G.C-AMENDMENT-2. The parameter-free value baseline, its LOSO gate, and the combined P x V
    policy re-run with it in place of the learned lift head. No new measurement is bought: the gains
    are the ones already banked and gated."""
    torch.manual_seed(SEED); np.random.seed(SEED)
    rng = np.random.default_rng(SEED)
    if args.threads:
        torch.set_num_threads(int(args.threads))
    t00 = time.time()

    tr, comp = scan_heaps()
    partition_identity_check(tr)
    from strategy_channel_suite import (build_greedy_env, contract_selftest, genv_mean,
                                        CERT_TOWER, CERT_CONC)
    from ndcg_tree import build_pool, leak_assert, branch_of_selftest
    E = build_greedy_env(args.snapshot, "validation", args.concept_ckpt, "meanvec/val")
    contract_selftest(E, E["rows"]); leak_assert(E, E["rows"], "meanvec/val")
    rows = E["rows"]
    f0, tl0 = genv_mean(E, [], rows)
    assert round(f0, 8) == VAL_M0_FULL, f"canonical snap FAILED: {f0!r}"
    log(f"[snap] validation m=0 intercept full={f0!r} tail={tl0!r} -- canonical, exact")
    pool, _ip, _cp = build_pool(E, 500, 500)
    branch_of_selftest(E, rows[:2000], [pool[0], pool[500], (1, 114)])
    ni, C = E["ctx"].ni, E["C"]
    catalogue = [(0, i) for i in range(ni)] + [(1, c) for c in range(C)]
    phi_t, norm = build_phi(E, catalogue)
    pool_cat = {"pool": [tuple(q) for q in pool],
                "index": {(0, i): i for i in range(ni)} | {(1, c): ni + c for c in range(C)}}

    t16 = json.load(open(TREE16_JSON))
    all_states, nver = banked_states(E, tr, t16, [tuple(q) for q in pool], args.verify_n, rng)
    states = [s for s in all_states if s["complete"]]
    assert len(states) == 7

    # ★ the extraction check the amendment asks for, stated as a number rather than a claim
    rootst = [s for s in states if s["nid"] == "root"][0]
    ra = max(rootst["gains"].items(), key=lambda kv: kv[1])
    log(f"[level0] root complete vector: {len(rootst['gains'])} candidates, argmax {list(ra[0])} at "
        f"gain {ra[1]:.8f} -- ndcgtree.json records splitter "
        f"{tr['nodes']['root']['splitter']} gain {tr['nodes']['root']['splitter_gain']:.8f}")
    assert list(ra[0]) == [int(x) for x in tr["nodes"]["root"]["splitter"]]
    assert abs(ra[1] - float(tr["nodes"]["root"]["splitter_gain"])) < 1e-12

    blob = torch.load(args.outcome_ckpt, map_location="cpu")
    assert not blob.get("prior_feat")
    om = OutcomeModel(blob["d_z"], blob["d_phi"], state=blob["state"], use_pos=blob["use_pos"],
                      d_k=blob["d_k"], d_h=blob["d_h"])
    om.load_state_dict(blob["model"]); om.eval()
    for q in om.parameters():
        q.requires_grad_(False)
    vt = np.load(os.path.join(TRAJ_DIR, "val_traj.npz"))
    Vv = {"uid": vt["uid"], "label_idx": vt["label_idx"], "outcome": vt["outcome"],
          "Z": np.load(os.path.join(TRAJ_DIR, "z_val.npy"), mmap_mode="r")}
    p2c = pool_to_catalogue([tuple(q) for q in pool], ni)
    packs = {s["nid"]: state_pack(s, Vv, phi_t, p2c, pool_cat, om) for s in states}

    # ---------- the baseline HARD RULE 2 asked for --------------------------------------------
    mv = mean_vector_loso(states, packs)
    rho = float(np.mean([r["spearman"] for r in mv]))
    log("=" * 100)
    log(f"[F.2 GATE / parameter-free mean] mean held-out-state Spearman {rho:+.4f} vs the bar "
        f"{RHO_GATE:+.2f} -> {'PASS' if rho >= RHO_GATE else 'FAIL'}; the LEARNED head scored "
        f"+0.0623 on the same folds. Reference: measured criterion {RHO_MEASURED_CRITERION:+.3f}, "
        f"training-loss surrogate {RHO_TRAINING_LOSS_SURROGATE:+.3f}")

    # ---------- the policy, with the mean vector as the value half -----------------------------
    pool_t = [tuple(q) for q in pool]
    gain, ncnt = mean_gain_table(states, packs, pool_t)
    rootpk = packs["root"]
    abar = np.zeros(len(pool_t), np.float32)
    with torch.no_grad():
        pa = (rootpk["P"][0] + rootpk["P"][1]).mean(0).numpy()      # cohort P(answered) per candidate
    idx = {q: i for i, q in enumerate(pool_t)}
    for j, q in enumerate(rootpk["cand"]):
        abar[idx[q]] = pa[j]
    log(f"[meanvec] static value table over {int((ncnt > 0).sum())}/{len(pool_t)} pool candidates; "
        f"cohort answer probability spans {abar[abar > 0].min():.3f}..{abar.max():.3f}")

    from policy_net import rollout
    from strategy_channel_suite import genv_eval
    cat_idx = {q: i for i, q in enumerate(catalogue)}
    phi_pool = np.ascontiguousarray(phi_t[[cat_idx[q] for q in pool_t]])
    sc = json.load(open(STATIC_COMBINED))
    sf, stl = genv_eval(E, [(int(a), int(b)) for a, b in sc["sequence"]], rows)
    assert abs(float(np.nanmean(sf)) - STATIC_VAL_FULL) < 1e-12, "BANKED-HEADLINE ASSERT FAILED"
    log(f"[banked] greedy_static_policyB_combined reproduces EXACTLY: full={float(np.nanmean(sf))!r}")
    si = json.load(open(STATIC_ITEMS))
    if_, itl = genv_eval(E, [(int(a), int(b)) for a, b in si["sequence"]], rows)
    bank_full = paired_bootstrap(sf - if_); bank_tail = paired_bootstrap(stl - itl)
    log(f"[F.4 BAR] bank-alone full {bank_full['mean']:+.5f} "
        f"[{bank_full['lo']:+.5f},{bank_full['hi']:+.5f}]")

    arms, peruser = {}, {}
    for nm, use_out in (("meanvec_PxV", True), ("meanvec_static", False)):
        pol = MeanVectorPolicy(om, gain, abar, use_outcome=use_out)
        t0 = time.time()
        ro = rollout(E, pol, phi_pool, pool_t, rows)
        f, t = ro["curves"][16]
        ans = (ro["outs"] != OUTCOME["noanswer"])
        arms[nm] = {"full": float(np.nanmean(f)), "tail": float(np.nanmean(t)),
                    "gain_vs_static_combined_full": paired_bootstrap(f - sf),
                    "gain_vs_static_combined_tail": paired_bootstrap(t - stl),
                    "n_distinct_interviews": len({tuple(r) for r in ro["asked"]}),
                    "channel_mix_concept_share": channel_mix(ro["asked"], pool_t),
                    "answer_rate": float(ans.mean()),
                    "uses_outcome_head": use_out,
                    "minutes": round((time.time() - t0) / 60, 2)}
        peruser[nm] = (f, t)
        g = arms[nm]["gain_vs_static_combined_full"]
        log(f"[{nm}] full {arms[nm]['full']:.5f} | gain vs combined static {g['mean']:+.5f} "
            f"[{g['lo']:+.5f},{g['hi']:+.5f}] | answer rate {arms[nm]['answer_rate']:.4f} | "
            f"concept share {arms[nm]['channel_mix_concept_share']:.4f} | "
            f"{arms[nm]['n_distinct_interviews']} distinct interviews")

    # ---------- ★ THE CEILING EXPERIMENT ------------------------------------------------------
    # Policy score as a function of VALUE-ESTIMATOR ACCURACY. The gate (rho >= +0.40) was set by
    # asking whether an estimator sits nearer the measured criterion (+0.784) than the training-loss
    # surrogate (-0.149). The mean vector PASSES it at +0.4163 and the policy built on it still loses
    # at -0.02911, so that bar was too low and we cannot currently state a finish line. This states it:
    # degrade the value vector to a controlled Spearman and watch the policy, several seeds per level.
    ceiling = []
    if not args.no_ceiling:
        levels = [float(x) for x in args.rho_levels.split(",")]
        for rho_t in levels:
            for sd in range(args.ceiling_seeds):
                v, rho_real = noisy_value_table(gain, rho_t, seed=SEED + 977 * sd)
                pol = MeanVectorPolicy(om, v, abar, use_outcome=True)
                t0 = time.time()
                ro = rollout(E, pol, phi_pool, pool_t, rows)
                f, t = ro["curves"][16]
                ans = (ro["outs"] != OUTCOME["noanswer"])
                g = paired_bootstrap(f - sf)
                ceiling.append({"rho_target": rho_t, "rho_realised": rho_real, "seed": sd,
                                "full": float(np.nanmean(f)), "tail": float(np.nanmean(t)),
                                "gain_vs_static_combined_full": g,
                                "answer_rate": float(ans.mean()),
                                "n_distinct_interviews": len({tuple(r) for r in ro["asked"]}),
                                "minutes": round((time.time() - t0) / 60, 2)})
                log(f"[ceiling rho={rho_real:.3f} seed{sd}] full {ceiling[-1]['full']:.5f} | "
                    f"gain vs combined static {g['mean']:+.5f} [{g['lo']:+.5f},{g['hi']:+.5f}] | "
                    f"answer rate {ceiling[-1]['answer_rate']:.4f}")
        # the rho = 1.0 anchor is the meanvec_PxV arm already measured above
        ceiling.append({"rho_target": 1.0, "rho_realised": 1.0, "seed": 0,
                        "full": arms["meanvec_PxV"]["full"],
                        "tail": arms["meanvec_PxV"]["tail"],
                        "gain_vs_static_combined_full":
                            arms["meanvec_PxV"]["gain_vs_static_combined_full"],
                        "answer_rate": arms["meanvec_PxV"]["answer_rate"],
                        "n_distinct_interviews": arms["meanvec_PxV"]["n_distinct_interviews"],
                        "minutes": arms["meanvec_PxV"]["minutes"]})
        log("=" * 100)
        log("[CEILING] policy gain vs combined static, by value-estimator accuracy "
            f"(bar: bank-alone {bank_full['mean']:+.5f}, tree {TREE_VAL_GAIN_FULL:+.5f})")
        for rt in sorted({c["rho_target"] for c in ceiling}):
            gs = [c["gain_vs_static_combined_full"]["mean"] for c in ceiling
                  if c["rho_target"] == rt]
            log(f"  rho={rt:.2f}  gain {np.mean(gs):+.5f}  [{min(gs):+.5f}, {max(gs):+.5f}] "
                f"over {len(gs)} seed(s)"
                + ("   <-- BEATS BANK-ALONE" if np.mean(gs) > bank_full["mean"] else ""))

    out = {"analysis": "meanvec_value_baseline", "design_sheet":
           "docs/design/PAPER_B_POLICY_DESIGN.md G.C-AMENDMENT-2", "citable": True,
           "test_touched": False, "split": "validation",
           "level0_root_argmax": list(ra[0]), "level0_root_gain": ra[1],
           "level0_root_n": len(rootst["gains"]),
           "meanvec_loso": mv, "meanvec_mean_spearman": rho,
           "learned_head_mean_spearman": 0.0623, "F2_gate_bar": RHO_GATE,
           "F2_gate_pass_meanvec": bool(rho >= RHO_GATE),
           "n_pool_candidates_with_value": int((ncnt > 0).sum()),
           "static_combined": {"full": float(np.nanmean(sf)), "tail": float(np.nanmean(stl))},
           "bank_alone_gain_full": bank_full, "bank_alone_gain_tail": bank_tail,
           "tree_val_gain_full": TREE_VAL_GAIN_FULL, "arms": arms,
           # ★ the one comparison that isolates what the OUTCOME head contributes inside a policy:
           # both arms carry the identical value table, so the only difference is whether the
           # per-user answer probability reweights it.
           "outcome_head_contribution_full": paired_bootstrap(
               peruser["meanvec_PxV"][0] - peruser["meanvec_static"][0]),
           "outcome_head_contribution_tail": paired_bootstrap(
               peruser["meanvec_PxV"][1] - peruser["meanvec_static"][1]),
           "ceiling": ceiling,
           "ceiling_note": ("policy score as a function of value-estimator accuracy. rho is measured "
                            "against the mean-of-7 static value table, so rho=1.0 is the best a STATIC "
                            "per-question value estimator can be -- not the true per-state optimum. If "
                            "the curve is still below the bank-alone bar at rho=1.0, no static value "
                            "estimator can make this design competitive and only a state-conditioned "
                            "one could -- which is the object that lost to the static table."),
           "bar_bank_alone": bank_full["mean"], "bar_tree": TREE_VAL_GAIN_FULL,
           "open_vocab_note": ("the mean-vector estimator is a LOOKUP TABLE over the 1,000-candidate "
                               "pool and has no value for a question outside it, so the open-vocabulary "
                               "arm is structurally unavailable to it. That is the scalability "
                               "argument for a parametric value head reappearing as a limitation of "
                               "the estimator that beats it."),
           "minutes": round((time.time() - t00) / 60, 2)}
    p = os.path.join(OUTDIR, "meanvec_value.json")
    with open(p + ".tmp", "w") as fh:
        json.dump(out, fh, indent=2, default=float)
    os.replace(p + ".tmp", p)
    np.savez_compressed(p.replace(".json", "_per_user.npz"),
                        pxv_full=peruser["meanvec_PxV"][0], pxv_tail=peruser["meanvec_PxV"][1],
                        static_full=peruser["meanvec_static"][0],
                        static_tail=peruser["meanvec_static"][1],
                        combined_static_full=sf, combined_static_tail=stl,
                        rows=np.asarray(rows, np.int64))
    c = out["outcome_head_contribution_full"]
    log(f"[outcome-head contribution] P x V minus the SAME value table without the outcome head: "
        f"{c['mean']:+.5f} [{c['lo']:+.5f},{c['hi']:+.5f}] full, excludes zero = {c['excludes_zero']}")
    log(f"[out] {p} ({out['minutes']:.1f}m)")
    return out


def run_allstates(args):
    r"""★ THE STATE-DIVERSITY EXPERIMENT (author's question: we used 7 states, there are far more).

    Refit the value head on EVERY banked state and evaluate LOSO on the **7 COMPLETE states only**, so
    the evaluation is never on a survivor-biased vector and the contrast isolates the effect of state
    diversity. Same rho >= +0.40 bar, same parameter-free +0.4163 reference, per-state as before.

    Excluding the level>0 material was right for a clean GATE test and wrong for TRAINING; the
    question is empirical. If the bias makes it worse, that is a clean finding about the bias; if it
    helps, label starvation is confirmed.
    """
    torch.manual_seed(SEED); np.random.seed(SEED)
    rng = np.random.default_rng(SEED)
    if args.threads:
        torch.set_num_threads(int(args.threads))
    t00 = time.time()

    allst = enumerate_all_banked_states()
    tr, _comp = scan_heaps()
    partition_identity_check(tr)

    from strategy_channel_suite import (build_greedy_env, contract_selftest, genv_mean,
                                        set_active_rows, CERT_TOWER, CERT_CONC)
    from ndcg_tree import build_pool, leak_assert, branch_of_selftest
    E = build_greedy_env(args.snapshot, "validation", args.concept_ckpt, "allstates/val")
    contract_selftest(E, E["rows"]); leak_assert(E, E["rows"], "allstates/val")
    rows_all = E["rows"]
    f0, tl0 = genv_mean(E, [], rows_all)
    assert round(f0, 8) == VAL_M0_FULL, f"canonical snap FAILED: {f0!r}"
    log(f"[snap] validation m=0 intercept full={f0!r} -- canonical, exact")
    pool, _ip, _cp = build_pool(E, 500, 500)
    branch_of_selftest(E, rows_all[:2000], [pool[0], pool[500], (1, 114)])
    ni, C = E["ctx"].ni, E["C"]
    catalogue = [(0, i) for i in range(ni)] + [(1, c) for c in range(C)]
    phi_t, norm = build_phi(E, catalogue)
    pool_cat = {"pool": [tuple(q) for q in pool],
                "index": {(0, i): i for i in range(ni)} | {(1, c): ni + c for c in range(C)}}
    d_z = E["ctx"].d

    t16 = json.load(open(TREE16_JSON))
    gated, nver = banked_states(E, tr, t16, [tuple(q) for q in pool], args.verify_n, rng)
    complete = [g for g in gated if g["complete"]]
    assert len(complete) == 7
    complete_keys = {(f"{None}", tuple(sorted(g["prefix"]))) for g in complete}

    blob = torch.load(args.outcome_ckpt, map_location="cpu")
    assert not blob.get("prior_feat")
    om = OutcomeModel(blob["d_z"], blob["d_phi"], state=blob["state"], use_pos=blob["use_pos"],
                      d_k=blob["d_k"], d_h=blob["d_h"])
    om.load_state_dict(blob["model"]); om.eval()
    for q_ in om.parameters():
        q_.requires_grad_(False)

    # ---- rows for every banked state: tree nodes by descent, everything else the full cohort ----
    from ndcg_tree import split_rows
    rows_of = {"root": [int(r) for r in rows_all]}

    def rows_for(nid):
        if nid in rows_of:
            return rows_of[nid]
        parent, _br = nid.rsplit(".", 1)
        prows = rows_for(parent)
        kids, _ar, _z = split_rows(E, prows, tuple(int(x) for x in tr["nodes"][parent]["splitter"]))
        for k, v in kids.items():
            rows_of[f"{parent}.{k}"] = v
        return rows_of[nid]

    packs, meta = {}, {}
    t0 = time.time()
    for i, (key, st) in enumerate(sorted(allst.items(), key=lambda kv: -len(kv[1]["gains"]))):
        node = st.get("node")
        if node and node in tr["nodes"]:
            rws = rows_for(node)
            if len(rws) != st["n_rows"]:
                log(f"[allstates] SKIP {node}@{st['level']}: cohort {len(rws)} != banked "
                    f"{st['n_rows']}")
                continue
        elif st["n_rows"] == len(rows_all):
            rws = rows_all
        else:
            log(f"[allstates] SKIP a state with {st['n_rows']} rows and no recoverable cohort")
            continue
        pk = build_state_pack_live(E, st, phi_t, pool_cat, om, ni, rws)
        if pk is None:
            continue
        name = f"{node or 'static'}@{st['level']}#{i}"
        packs[name] = pk
        meta[name] = {"n_users": len(rws), "n_cand": len(pk["cand"]), "level": st["level"],
                      "node": node, "src": st["src"][:2]}
        if (i + 1) % 40 == 0:
            log(f"[allstates] packed {i + 1}/{len(allst)} states ({(time.time() - t0) / 60:.1f}m)")
    set_active_rows(E, rows_all)
    log(f"[allstates] {len(packs)} training states packed in {(time.time() - t0) / 60:.1f}m; "
        f"{sum(m['n_cand'] for m in meta.values()):,} training triples")

    # ---- map the 7 evaluation states onto their packs -------------------------------------------
    eval_names = {}
    for g in complete:
        want = tuple(sorted([tuple(q) for q in g["prefix"]]))
        hits = [nm for nm, pk in packs.items()
                if meta[nm]["node"] == g["nid"] and meta[nm]["level"] == 0]
        assert hits, f"no pack found for the complete state {g['nid']}"
        eval_names[g["nid"]] = hits[0]
    log(f"[allstates] evaluation states (COMPLETE only): "
        + ", ".join(f"{k}->{v}" for k, v in eval_names.items()))

    # ---- LOSO over the 7 complete states, training on everything else --------------------------
    seeds = [SEED + 1000 * i for i in range(args.lift_seeds)]
    loso = []
    for nid, evname in eval_names.items():
        pk = packs[evname]
        trp = [p_ for nm, p_ in packs.items() if nm != evname]
        per_seed = []
        for sd in seeds:
            m, h, bep = fit_lift(trp, d_z, phi_t.shape[1], args, f"all{len(trp)}-{nid}-s{sd}",
                                 seed=sd)
            c = composed(m, pk, offset=args.v_offset).numpy()
            rk = int((pk["target_gain"] > pk["target_gain"][int(np.argmax(c))]).sum()) + 1
            per_seed.append({"seed": sd, "best_epoch": bep,
                             "spearman": spearman(c, pk["target_gain"]),
                             "rank_of_argmax": rk})
        rho = [x["spearman"] for x in per_seed]
        loso.append({"held_out_state": nid, "n_train_states": len(trp),
                     "n_candidates": len(pk["cand"]),
                     "spearman": float(np.mean(rho)), "spearman_min": float(np.min(rho)),
                     "spearman_max": float(np.max(rho)), "per_seed": per_seed,
                     "rank_of_argmax": float(np.mean([x["rank_of_argmax"] for x in per_seed]))})
        log(f"[LOSO/all-states {nid}] {len(trp)} training states, {len(seeds)} seeds -> "
            f"held-out Spearman {loso[-1]['spearman']:+.4f} "
            f"[{loso[-1]['spearman_min']:+.4f}, {loso[-1]['spearman_max']:+.4f}] | "
            f"mean argmax rank {loso[-1]['rank_of_argmax']:.0f}/{len(pk['cand'])}")
    rho_mean = float(np.mean([r["spearman"] for r in loso]))
    log("=" * 100)
    log(f"[STATE DIVERSITY] {len(packs)} training states -> mean held-out Spearman {rho_mean:+.4f}")
    log(f"[STATE DIVERSITY]   7 clean states  : +0.0623   (the gate run)")
    log(f"[STATE DIVERSITY]  22 states        : +0.0819")
    log(f"[STATE DIVERSITY] {len(packs):3d} states        : {rho_mean:+.4f}")
    log(f"[STATE DIVERSITY] parameter-free mean of 6 vectors: +0.4163 -- the bar to beat, and the "
        f"gate is {RHO_GATE:+.2f}")

    out = {"analysis": "lift_head_all_banked_states", "design_sheet":
           "docs/design/PAPER_B_POLICY_DESIGN.md PART F", "citable": True, "test_touched": False,
           "split": "validation",
           "n_states_total": len(allst), "n_states_trained": len(packs),
           "n_training_triples": int(sum(m["n_cand"] for m in meta.values())),
           "excluded_banks": [m for m in SANITY_MARKERS],
           "loso": loso, "loso_mean_spearman": rho_mean,
           "reference_7_states": 0.0623, "reference_22_states": 0.0819,
           "reference_parameter_free_mean": 0.4163,
           "F2_gate_bar": RHO_GATE, "F2_gate_pass": bool(rho_mean >= RHO_GATE),
           "state_meta": meta, "lift_seeds": args.lift_seeds, "lift_epochs": args.lift_epochs,
           "bias_note": ("every level>0 entry is a CELF SURVIVOR -- the high-gain candidates were "
                         "popped first, so these states are enriched for LOW gains. That is "
                         "truncation on the dependent variable. Evaluation is therefore only ever "
                         "on the 7 COMPLETE, unbiased states."),
           "minutes": round((time.time() - t00) / 60, 2)}
    pth = os.path.join(OUTDIR, "lift_head_allstates.json")
    with open(pth + ".tmp", "w") as fh:
        json.dump(out, fh, indent=2, default=float)
    os.replace(pth + ".tmp", pth)
    log(f"[out] {pth} ({out['minutes']:.1f}m)")
    return out


def main():
    ap = argparse.ArgumentParser(description="Part F lift head + combined P x V policy")
    ap.add_argument("--no_ceiling", action="store_true",
                    help="skip the value-estimator-accuracy sweep")
    ap.add_argument("--rho_levels", default="0.5,0.6,0.7,0.8,0.9,0.95")
    ap.add_argument("--ceiling_seeds", type=int, default=3,
                    help="noise draws per accuracy level; a single draw is a point estimate of a "
                         "noisy thing and this decision must not rest on one")
    ap.add_argument("--allstates", action="store_true",
                    help="refit the value head on EVERY banked state; evaluate LOSO on "
                         "the 7 COMPLETE states only")
    ap.add_argument("--meanvec", action="store_true",
                    help="G.C-AMENDMENT-2: the parameter-free mean-vector value "
                         "baseline, its LOSO gate, and the P x V policy re-run with it")
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
    ap.add_argument("--no_v_offset", action="store_true",
                    help="predict V free-form instead of as base(s) + delta. The offset form is "
                         "the default: the level is a known per-state constant (the heap base, "
                         "gated exact by G-L1), so fitting it from seven examples spends the "
                         "label budget on something already on disk.")
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
    args.v_offset = not args.no_v_offset
    if args.allstates:
        from strategy_channel_suite import CERT_TOWER as _T2, CERT_CONC as _C2
        args.snapshot = args.snapshot or _T2
        args.concept_ckpt = args.concept_ckpt or _C2
        assert args.i_have_author_goahead, 'REFUSING to run without the go-ahead.'
        return run_allstates(args)
    if args.meanvec:
        from strategy_channel_suite import CERT_TOWER as _T, CERT_CONC as _C
        args.snapshot = args.snapshot or _T
        args.concept_ckpt = args.concept_ckpt or _C
        assert args.i_have_author_goahead, 'REFUSING to run without the go-ahead.'
        return run_meanvec(args)
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
