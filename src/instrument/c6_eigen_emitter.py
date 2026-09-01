r"""PAPER C / C6 -- IS THE MEASURED-BEST PROBE DIRECTION THE COHORT'S PRINCIPAL DISAGREEMENT AXIS?

THE HYPOTHESIS UNDER TEST (author, 2026-09-01)
----------------------------------------------
At an interview state `s` the cohort `C(s)` is the set of users consistent with the answers so far.
A question's measured NDCG gain comes from splitting the RESIDUAL DISAGREEMENT inside that cohort.
Under a linear-Gaussian reading of elicitation the most informative probe direction is approximately
the PRINCIPAL AXIS of the cohort's residual covariance, `v1(s)`. That is computable in CLOSED FORM
at every state -- no training, no measured gain table. C1-final's emitter is a learned regression
onto measured-gain blends that exists at 72 states and does not extrapolate (`C1_FINAL.md`).

    If |cos(v1(s), blend_target(s))| is high, Paper C gains a closed-form emitter that works
    everywhere and the generalisation problem dissolves.

This file MEASURES that. It trains nothing.

WHAT IS IMPORTED, NEVER REIMPLEMENTED
-------------------------------------
  `c1_final.load_sweeps`         the 72 FULL exact whole-pool gain tables + their contract check
  `c1_pilot.build_targets`       the top-m gain-weighted blend (the ingredients AND their weights)
  `c4d_transcript.build_m7dj_bank`  m7dj's [k(phi), b(phi)] bank, so the blend can be gated
                                 BIT-IDENTICALLY against c1_final's banked targets before anything
                                 downstream is computed
  `c4_pilot.reference_directions`   R_s = sum_j w_j tower(q_j) -- the blend target expressed in the
                                 TOWER space, the object C4 already uses for exactly this purpose
  `ndcg_tree.branch_of`          the tree's own three-way answer rule, for the cohort descent

★ THE TWO CONVENTIONS THIS FILE IS ACCOUNTABLE FOR (C0's common-mode lesson)
----------------------------------------------------------------------------
1. SPACE. Everything lives in the 200-d TOWER latent space, POPULARITY-STRIPPED exactly as
   `c3b_adapter.TowerBridge` does it: an item's vector is `Ew[i] = (Wd[i] - mean(Wd)) - <.,u1>u1`,
   a concept's is `d_c[c]` (that same quantity averaged over the concept's members, unit length),
   and a USER's latent is the tower fold with the SAME `u1` component removed. The bank mean
   `mean(Wd)` is therefore removed from every question-side vector by construction.
2. MEAN-REMOVAL. An eigenvector of a covariance is a mean-removed object by definition (the
   covariance is taken about the cohort's OWN mean). The target side is bank-mean-removed as above.
   Every cosine reported as the headline is therefore MEAN-REMOVED ON BOTH SIDES. The raw
   (bank-mean-retained, i.e. `Wd` rows uncentred) variant is banked beside it as a robustness row,
   because C0/C4 have now been bitten three times by a raw-vs-mean-removed reading.
   ⚠ SIGN. An eigenvector's sign is arbitrary. The HEADLINE is |cos|. A signed value is also
   reported under the declared convention "v1 points to the side of the cohort's projection
   distribution that carries the heavier tail" (positive skew of <z - zbar, v1>).

COHORT RESOLUTION -- A DECLARED DEPARTURE FROM `c1_final.py`, AND WHY
--------------------------------------------------------------------
`c1_final` resolves a state's cohort as the (user, position) cells of the 85,289-user COLD
TRAJECTORY BANK. This file uses the TREE'S OWN FIT COHORT instead: the users among the 10,000 TRAIN
users the tree was grown on who reach that node, re-derived here with `ndcg_tree.branch_of` and
asserted against every node's banked `n_users` and `child_sizes`. The reason is not convenience: the
gain tables the target is built from were MEASURED on that cohort (`C1_FINAL_SCOPING.md` 1.2 --
sweep `n_rows` == node `n_users` at 72/72, pinned by `rows_sha1`). A covariance computed over a
different population than the gains would be comparing two populations' geometry and calling the
difference a result. The bank cohort is additionally an anchor-selected 60.6% of train.

TWO USER REPRESENTATIONS ARE MEASURED, because the phrase "the cohort's user latents" is ambiguous
in a COLD protocol and the ambiguity is load-bearing:
  * `profile` (PRIMARY) -- the tower fold of the user's whole FOLD-IN profile (`E["lvl"]`, the same
    graded lookup the tree's answer rule reads). This is the user's full-information latent: what
    the interview is trying to discover, hence what "residual disagreement" must mean.
  * `belief`  (SECONDARY) -- `genv_latent(E, prefix, cohort)`, the latent the scorer actually holds
    at that state. In a cold interview this is a function of the answers to the PREFIX only, so its
    within-cohort spread is the sub-branch variation of those same answers and carries no
    information about anything unasked. Reported so the distinction is visible, not assumed.

FIREWALL. TRAIN-side states only, the tree's own 10,000-user train slice. Nothing here can build a
test environment; the 10,000 canonical TEST users are not built, not loaded, not touched. Nothing is
trained and no checkpoint is written.
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)
if "--threads" not in sys.argv:                    # set BEFORE importing c1_* (they append their own)
    sys.argv += ["--threads", "3"]
_NT = str(min(int(sys.argv[sys.argv.index("--threads") + 1]), 4))
os.environ["OMP_NUM_THREADS"] = _NT
os.environ.setdefault("MKL_NUM_THREADS", _NT)
os.environ.setdefault("OPENBLAS_NUM_THREADS", _NT)

import argparse
import json
import time

import numpy as np
import torch

from policy_net import log
from c0_geometry import PAPERC, N_BANK, git, git_block, savez_atomic, dist
import c1_pilot as C1
import c1_final as C1F
import c4_pilot as C4
import c4d_transcript as C4D

SEED = 98765
N_BOOT = 10000
M = 5                                   # the registered blend width
N_SEARCHED = 72
D_TOWER = 200
C1F_NPZ = os.path.join(PAPERC, "c1_final_per_state.npz")
C1F_JSON = os.path.join(PAPERC, "c1_final.json")
OUT_JSON = os.path.join(PAPERC, "c6_eigen.json")
OUT_NPZ = os.path.join(PAPERC, "c6_eigen_per_state.npz")
# C1-final's registered held-out cell and the number this run is read against.
HELD_CELL = "primary_root_like_subtree"
C1F_HELDOUT_COS = -0.11948


# ============================================================================ provenance
def import_scoped_git_block(args):
    """`c0_geometry.git_block`'s HARD RULE 10 refusal, scoped to the modules THIS run imports.

    Verbatim in construction from `c4c_decoding.import_scoped_git_block` (the same working tree, the
    same two Paper-B files dirty in another workstream). The repo-wide dirty list is recorded in
    full; the refusal is re-applied over the intersection with `sys.modules`.
    """
    GB = git_block(allow_dirty=True, threads=args.threads,
                   extra={"status_short": git("status", "--short")})
    imported = set()
    for m in list(sys.modules.values()):
        f = getattr(m, "__file__", None)
        if f and os.path.abspath(f).replace("\\", "/").find("/src/instrument/") >= 0:
            imported.add(os.path.basename(f))
    offend = [ln for ln in GB["src_instrument_dirty"]
              if os.path.basename(ln.replace("\\", "/").split()[-1]) in imported]
    GB["guard"] = ("HARD RULE 10, IMPORT-SCOPED: refuses if any src/instrument file this run "
                   "IMPORTS is dirty. The repo-wide dirty list is recorded in full above.")
    GB["n_modules_imported_from_src_instrument"] = len(imported)
    GB["dirty_src_instrument_files_imported_by_this_run"] = offend
    GB["src_instrument_clean_for_this_run"] = not offend
    if offend and not args.allow_dirty:
        raise SystemExit("REFUSING TO RUN (HARD RULE 10, import-scoped): this run imports "
                         "uncommitted src/instrument code --\n  " + "\n  ".join(offend))
    return GB


# ============================================================================ the tower side
class TowerSide:
    r"""`c3b_adapter.TowerBridge`'s QUESTION-side construction, on this run's env.

    TowerBridge hard-wires `split="validation"`; this run's cohort is the tree's TRAIN slice, so the
    six lines that build `Ew` / `d_c` / `u1` are taken here instead of instantiating a second
    environment. That is legitimate for a specific, checkable reason: all three are functions of the
    frozen decoder alone -- `u1` is the top PCA direction of `Wd - mean(Wd)`
    (`tradeoff_ledger.build_shared`), `d_c` is `build_concept_dirs(ctx.Wd, members, tags)`, and `Wd`
    is the certified tower's decoder weight. No cohort statistic enters any of them.

    Not taken on trust: TowerBridge's OWN control is re-run here -- averaging `Ew` over each
    concept's members must reproduce the env's `d_c` to float tolerance.
    """

    def __init__(self, E):
        ctx, sh = E["ctx"], E["sh"]
        self.E = E
        self.ni, self.C = int(ctx.ni), int(E["C"])
        Wd = np.asarray(ctx.Wd.detach().numpy(), np.float64)
        u1 = np.asarray(sh["u1"].detach().numpy(), np.float64)
        u1 = u1 / np.linalg.norm(u1)
        Ec = Wd - Wd.mean(0, keepdims=True)
        self.Wd = Wd
        self.Wd_mean = Wd.mean(0)
        self.Ew = Ec - np.outer(Ec @ u1, u1)               # item targets, centred + pop-stripped
        self.Ew_raw = Wd - np.outer(Wd @ u1, u1)           # the RAW (uncentred) robustness variant
        self.u1 = u1
        self.d_c = np.asarray(sh["d_c"].detach().numpy(), np.float64)
        tags, members = list(sh["tags"]), sh["members"]
        worst = 0.0
        d_c_raw = np.zeros_like(self.d_c)
        for i, t in enumerate(tags):
            mem = np.asarray(members[t], np.int64)
            v = self.Ew[mem].mean(0)
            d_c_raw[i] = self.Ew_raw[mem].mean(0)
            v = v / max(np.linalg.norm(v), 1e-12)
            worst = max(worst, float(np.abs(np.abs(v @ self.d_c[i]) - 1.0)))
        assert worst < 1e-3, (f"the reconstructed whitening does not reproduce the env's concept "
                              f"directions (worst 1-|cos| {worst:g})")
        self.d_c_raw = d_c_raw
        self.dc_reproduction_worst = worst
        assert self.Ew.shape[1] == D_TOWER and self.d_c.shape[1] == D_TOWER
        log(f"[tower] item targets Ew {self.Ew.shape}; concept directions reproduced from them "
            f"(worst 1-|cos| {worst:g}); ||u1||=1, ||mean(Wd)||={np.linalg.norm(self.Wd_mean):.4f}")

    def target_tower(self, bank_idx):
        return np.stack([self.Ew[int(i)] if int(i) < self.ni else self.d_c[int(i) - self.ni]
                         for i in bank_idx])

    def bank(self, raw=False):
        """The whole 19,390-entry question bank as tower vectors, items then concepts."""
        return (np.concatenate([self.Ew_raw, self.d_c_raw], 0) if raw else
                np.concatenate([self.Ew, self.d_c], 0))

    def strip(self, Z):
        """Popularity-strip a block of user latents, the same operation applied to the question side."""
        Z = np.asarray(Z, np.float64)
        return Z - np.outer(Z @ self.u1, self.u1)


# ============================================================================ the fit cohorts
def fit_cohorts(E, nodes, rows):
    """Re-derive every node's OWN fit cohort by descending the tree with `ndcg_tree.branch_of`.

    Contract-checked at every node, not at the root: the derived cohort size must equal the node's
    banked `n_users`, and the three derived child sizes must equal the banked `child_sizes`. A size
    match alone is what memory `cross-population-index-match-is-silent` warns about, so both are
    asserted at all 144 nodes -- a wrong branch order or a stale tree cannot pass.
    """
    from ndcg_tree import branch_of
    coh = {}
    stack = [("root", np.asarray(rows, np.int64))]
    checks = {"n_nodes_checked": 0, "n_users_mismatch": 0, "child_size_mismatch": 0,
              "n_users_falling_off": 0}
    while stack:
        nid, idx = stack.pop()
        nd = nodes[nid]
        coh[nid] = idx
        checks["n_nodes_checked"] += 1
        assert len(idx) == int(nd["n_users"]), (
            f"{nid}: derived cohort {len(idx)} != banked n_users {nd['n_users']}")
        if not nd.get("split"):
            continue
        lab, _ar, _nz = branch_of(E, idx, tuple(nd["splitter"]))
        cs = nd.get("child_sizes") or {}
        for name in ("like", "dislike", "noanswer"):
            sel = idx[lab == name]
            if name in cs:
                assert len(sel) == int(cs[name]), (
                    f"{nid}.{name}: derived {len(sel)} users, banked child_sizes {cs[name]}")
            child = f"{nid}.{name}"
            if child in nodes:
                if len(sel):
                    stack.append((child, sel))
                else:
                    coh[child] = sel
            else:
                checks["n_users_falling_off"] += int(len(sel))
    assert len(coh) == len(nodes), f"descended {len(coh)} nodes of {len(nodes)}"
    return coh, checks


# ============================================================================ latents
def profile_latents(E, rows, batch=500):
    """The tower fold of each user's WHOLE fold-in profile -- their full-information latent.

    `E["lvl"][r]` is the graded fold-in lookup (held-out targets excluded by construction), and it is
    the SAME object `ndcg_tree.branch_of` reads to decide a user's answer, so this is the user's own
    evidence and nothing else. The fold is `concept_fold.fold_items_frozen`, i.e. the tower's own
    encoder, called exactly as `strategy_channel_suite._env_item_state` calls it.
    """
    from concept_fold import fold_items_frozen
    lvl = E["lvl"]
    seqs = []
    for r in rows:
        d = lvl[int(r)]
        ids = np.fromiter(d.keys(), np.int64, len(d))
        lv = np.fromiter(d.values(), np.int64, len(d))
        seqs.append((ids, lv))
    Z = fold_items_frozen(E["ctx"].enc, seqs, batch=batch)
    return np.asarray(Z.numpy(), np.float64)


def eig_top(Z, k=3):
    """Eigen-decomposition of the cohort's covariance ABOUT ITS OWN MEAN.

    Sign convention (declared, and arbitrary either way): each eigenvector is oriented so the
    cohort's projection onto it has NON-NEGATIVE skew -- "positive on the side carrying the heavier
    tail of the cohort's spread". The headline statistic is |cos| and does not use this.
    """
    Zc = Z - Z.mean(0, keepdims=True)
    n = Zc.shape[0]
    Cv = (Zc.T @ Zc) / max(n - 1, 1)
    w, V = np.linalg.eigh(Cv)
    o = np.argsort(-w)
    w = w[o]
    V = V[:, o]
    for j in range(V.shape[1]):
        p = Zc @ V[:, j]
        sd = p.std()
        sk = float(((p / max(sd, 1e-30)) ** 3).mean()) if sd > 0 else 0.0
        if sk < 0:
            V[:, j] = -V[:, j]
    return w, V[:, :k], Zc


def unit(x, axis=-1):
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(n, 1e-30)


def boot_median(x, n_boot=N_BOOT, seed=SEED):
    x = np.asarray(x, np.float64)
    x = x[np.isfinite(x)]
    rng = np.random.default_rng(seed)
    bs = np.median(x[rng.integers(0, len(x), size=(n_boot, len(x)))], axis=1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"median": float(np.median(x)), "lo": float(lo), "hi": float(hi),
            "q1": float(np.percentile(x, 25)), "q3": float(np.percentile(x, 75)),
            "mean": float(x.mean()), "n": int(len(x))}


def boot_paired(a, b, n_boot=N_BOOT, seed=SEED):
    """Paired percentile bootstrap on the MEDIAN of the per-state difference (never the unpaired
    1.96*sqrt(2)*SE form -- memory `paired-test-kills-the-tie-claim`)."""
    d = np.asarray(a, np.float64) - np.asarray(b, np.float64)
    d = d[np.isfinite(d)]
    rng = np.random.default_rng(seed)
    bs = np.median(d[rng.integers(0, len(d), size=(n_boot, len(d)))], axis=1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"median_diff": float(np.median(d)), "mean_diff": float(d.mean()),
            "lo": float(lo), "hi": float(hi), "excludes_zero": bool(lo > 0 or hi < 0),
            "n_states_positive": int((d > 0).sum()), "n": int(len(d))}


def ndcg_at_k(order, gain, k=5):
    """NDCG@k of a candidate ORDER against the measured gains as graded relevance."""
    rel = np.maximum(np.asarray(gain, np.float64), 0.0)
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = float((rel[order[:k]] * disc[:len(order[:k])]).sum())
    ideal = np.sort(rel)[::-1][:k]
    idcg = float((ideal * disc[:len(ideal)]).sum())
    return dcg / idcg if idcg > 0 else np.nan


def spearman(a, b):
    from scipy.stats import spearmanr
    r = spearmanr(a, b)
    return float(r[0]), float(r[1])


# ============================================================================ the run
def run(args):
    from strategy_channel_suite import build_greedy_env, contract_selftest, genv_mean, genv_latent
    from ndcg_tree import build_pool, branch_of_selftest
    t00 = time.time()
    os.makedirs(PAPERC, exist_ok=True)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    GB = import_scoped_git_block(args)
    log(f"[hardrule10] rev {GB['rev'][:8]}; imported src/instrument modules "
        f"{GB['n_modules_imported_from_src_instrument']}; dirty AND imported "
        f"{len(GB['dirty_src_instrument_files_imported_by_this_run'])}")
    torch.set_num_threads(args.threads)

    # ---- the env: the tree's OWN train slice ---------------------------------------------------
    E = build_greedy_env(args.snapshot, "train", args.concept_ckpt, "paperC/c6",
                         cohort_slice=(0, args.n_users))
    contract_selftest(E, E["rows"])
    rows = np.asarray(E["rows"], np.int64)
    assert len(rows) == args.n_users, f"{len(rows)} cohort users, expected {args.n_users}"
    f0, t0_ = genv_mean(E, [], list(rows))
    assert f0 == C1F.TRAIN_M0_FULL, (
        f"CANONICAL SNAP FAILED: train m=0 full {f0!r}, banked {C1F.TRAIN_M0_FULL!r}")
    log(f"[gate] canonical TRAIN m=0 snap full={f0!r} tail={t0_!r} -- full asserted exact")

    pool, _ip, _cp = build_pool(E, 500, 500)
    pool = [(int(a), int(b)) for a, b in pool]
    pool_index = {q: i for i, q in enumerate(pool)}
    n_pool = len(pool)
    branch_of_selftest(E, rows[:2000], [pool[0], pool[500], (1, 114)])
    ni, C = int(E["ctx"].ni), int(E["C"])
    assert ni + C == N_BANK
    full_pool = [(0, i) for i in range(ni)] + [(1, c) for c in range(C)]
    pool2full = np.asarray([(i if ch == 0 else ni + i) for ch, i in pool], np.int64)

    br = TowerSide(E)

    # ---- the tree, the 72 exact gain tables, the blend ingredients ------------------------------
    tree = json.load(open(C1F.TREE_JSON, encoding="utf-8"))
    assert tree["split"] == "train" and int(tree["total_L"]) == C1F.TOTAL_L
    assert tree["test_touched"] is False
    nodes = tree["nodes"]
    states, gain_chk = C1F.load_sweeps(nodes, pool_index)
    assert len(states) == N_SEARCHED
    sids = [s["sid"] for s in states]

    m7 = C4D.build_m7dj_bank(E, full_pool, pool, C1F.WARM)
    Tt, TKt, meta = C1.build_targets(states, m7["Epool"], m7["Kpool"], pool_index, M)
    zf = np.load(C1F_NPZ, allow_pickle=True)
    assert [str(x) for x in zf["state_sid"]] == sids, "not c1_final's banked states"
    d_tgt = float(np.abs(Tt - np.asarray(zf[f"target_m{M}"], np.float32)).max())
    assert d_tgt == 0.0, (
        f"★ GATE FAILED: the m={M} blend rebuilt here is not the one c1_final banked "
        f"(max|diff| {d_tgt:g}). The ingredients/weights this run reads are not C1-final's. STOP.")
    log(f"[gate] the {N_SEARCHED} blend targets rebuilt here are BIT-IDENTICAL to c1_final's "
        f"(max|diff| {d_tgt:g})")

    # the blend target expressed in the TOWER space (C4's own object, imported)
    R, r_disc = C4.reference_directions(br, meta, pool, pool2full, n_states=N_SEARCHED)
    Ru = unit(R)
    log(f"[target] tower-space blend targets R {R.shape}; {r_disc['n_item_ingredients']} item / "
        f"{r_disc['n_concept_ingredients']} concept ingredients over {N_SEARCHED} states")
    iu = np.triu_indices(N_SEARCHED, 1)
    tgt_pair = np.abs((Ru @ Ru.T)[iu])

    # the answerable question bank, in tower space
    Qbank = br.bank()                                   # (19390, 200)
    Qbank_u = unit(Qbank)
    Qpool_u = Qbank_u[pool2full]                        # the 1,000 ANSWERABLE candidates
    assert Qpool_u.shape == (n_pool, D_TOWER)

    # the answer-rate ("popularity") prior over the pool, on the ROOT cohort -- a state-blind prior
    lvl = E["lvl"]
    ans_rate = np.zeros(n_pool)
    for j, (ch, cid) in enumerate(pool):
        if ch == 0:
            ans_rate[j] = float(np.mean([cid in lvl[int(r)] for r in rows]))
        else:
            ans_rate[j] = float(E["foldable"][rows, cid].mean())

    # ---- the fit cohorts -------------------------------------------------------------------------
    coh, coh_checks = fit_cohorts(E, nodes, rows)
    log(f"[cohorts] descended {coh_checks['n_nodes_checked']} nodes; every node's size and every "
        f"child size match the banked tree; {coh_checks['n_users_falling_off']} users land in an "
        f"unfitted branch")
    csize = np.asarray([len(coh[s]) for s in sids])
    assert (csize == np.asarray([s["n_fit_users"] for s in states])).all()

    # ---- the profile latents, folded ONCE for the whole cohort ----------------------------------
    log("[latent] folding the full fold-in profile of every cohort user through the tower")
    Zprof_all = profile_latents(E, rows)
    pos_of = {int(r): i for i, r in enumerate(rows)}
    Zprof_all_s = br.strip(Zprof_all)
    root_mean_prof = Zprof_all_s.mean(0)
    log(f"[latent] profile latents {Zprof_all_s.shape}; ||root mean|| "
        f"{np.linalg.norm(root_mean_prof):.4f}; mean ||z|| "
        f"{np.linalg.norm(Zprof_all_s, axis=1).mean():.4f}")

    out = {"analysis": "paperC_C6_eigen_emitter", "citable": True, "test_touched": False,
           "split": "train_states", "teacher": os.path.basename(C1F.TREE_JSON),
           "hypothesis": ("the measured-best probe direction at a state is approximately the "
                          "principal axis of the cohort's residual covariance"),
           "git": GB, "seed": SEED, "n_boot": N_BOOT,
           "conventions": {
               "space": ("200-d TOWER latent space, popularity-stripped exactly as "
                         "c3b_adapter.TowerBridge: item = (Wd - mean(Wd)) - <.,u1>u1, concept = d_c, "
                         "user latent = tower fold with the same u1 component removed"),
               "mean_removal": ("MEAN-REMOVED ON BOTH SIDES. Question side: the bank mean mean(Wd) "
                                "is removed by construction. Cohort side: the covariance is taken "
                                "about the cohort's OWN mean, so v1..v3 are mean-removed objects. "
                                "A raw (uncentred Wd) robustness row is banked beside the headline."),
               "sign": ("|cos| is the headline because an eigenvector's sign is arbitrary. The "
                        "signed value uses the declared convention: v_j is oriented so the cohort's "
                        "projection <z - zbar, v_j> has non-negative skew (positive on the side "
                        "carrying the heavier tail of the cohort's spread)."),
               "cohort": ("the tree's OWN fit cohort at each node -- the users among the 10,000 "
                          "TRAIN users the tree was grown on who reach that node, re-derived with "
                          "ndcg_tree.branch_of and asserted against every banked n_users AND every "
                          "banked child_sizes. This DEPARTS from c1_final's bank-cohort resolution "
                          "because the gain tables were measured on this cohort (sweep n_rows == "
                          "node n_users at 72/72)."),
               "latent": ("PRIMARY = `profile`, the tower fold of the user's whole fold-in profile "
                          "(their full-information latent). SECONDARY = `belief`, genv_latent at "
                          "the state's prefix (what the scorer holds; in a cold interview it is a "
                          "function of the prefix answers only).")},
           "gates": {"canonical_snap_train_m0_full": {"got": f0, "expected": C1F.TRAIN_M0_FULL,
                                                      "pass": True},
                     "canonical_snap_train_m0_tail": t0_,
                     "sweep_argmax_equals_splitter_max_abs_gain_err": gain_chk,
                     "blend_target_bit_identical_to_c1_final": d_tgt,
                     "tower_dc_reproduction_worst_1_minus_abscos": br.dc_reproduction_worst,
                     "cohort_descent": coh_checks},
           "target_tower_disclosure": r_disc,
           "target_diversity_tower_space": {
               "unit": "the 2,556 unordered pairs of the 72 tower-space blend targets",
               "abs_pairwise_cos": dist(tgt_pair),
               "note": ("the tower analogue of C5's D1. It is what makes the state-blind "
                        "leave-one-out null strong, and it must be read beside the headline.")},
           "states": {"n": N_SEARCHED, "fit_cohort_size": dist(csize),
                      "n_below_500_users": int((csize < 500).sum())},
           "readings": {}}
    npz = {"state_sid": np.asarray(sids), "state_depth": np.asarray([s["depth"] for s in states]),
           "cohort_size": csize, "R_tower": R}

    def flush():
        out["minutes"] = round((time.time() - t00) / 60, 2)
        with open(OUT_JSON + ".tmp", "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=1, default=float)
        os.replace(OUT_JSON + ".tmp", OUT_JSON)
        savez_atomic(OUT_NPZ, **npz)

    flush()

    # ================================================================= per-state geometry
    rng = np.random.default_rng(SEED)
    arms = {}
    for arm in ("profile", "belief"):
        log("#" * 100)
        log(f"# ARM {arm}")
        V1 = np.zeros((N_SEARCHED, D_TOWER))
        V3 = np.zeros((N_SEARCHED, 3, D_TOWER))
        LAM = np.zeros((N_SEARCHED, 10))
        lam1_frac = np.zeros(N_SEARCHED)
        eff_rank = np.zeros(N_SEARCHED)
        MU = np.zeros((N_SEARCHED, D_TOWER))          # cohort mean MINUS the root mean
        MUraw = np.zeros((N_SEARCHED, D_TOWER))       # the cohort mean itself
        halfcos = np.zeros(N_SEARCHED)
        # the arm's own reference point for the "cohort mean direction" null, computed ONCE
        base = (root_mean_prof if arm == "profile" else
                br.strip(genv_latent(E, [], list(rows)).numpy().astype(np.float64)).mean(0))
        t0 = time.time()
        for i, s in enumerate(states):
            idx = coh[s["sid"]]
            if arm == "profile":
                Z = Zprof_all_s[np.asarray([pos_of[int(r)] for r in idx], np.int64)]
            else:
                Z = br.strip(genv_latent(E, s["prefix"], list(idx)).numpy().astype(np.float64))
            w, V, Zc = eig_top(Z, 3)
            LAM[i] = w[:10]
            lam1_frac[i] = float(w[0] / max(w.sum(), 1e-30))
            p = w / max(w.sum(), 1e-30)
            p = p[p > 0]
            eff_rank[i] = float(np.exp(-(p * np.log(p)).sum()))
            V1[i] = V[:, 0]
            V3[i] = V.T
            MUraw[i] = Z.mean(0)
            MU[i] = Z.mean(0) - base
            # split-half stability of v1 -- the control that decides whether v1 means anything at
            # cohorts of 200-10,000 users in 200 dimensions
            hr = np.random.default_rng(SEED + 7919 + i).permutation(len(Z))
            h = len(Z) // 2
            _wa, Va, _ = eig_top(Z[hr[:h]], 1)
            _wb, Vb, _ = eig_top(Z[hr[h:2 * h]], 1)
            halfcos[i] = abs(float(Va[:, 0] @ Vb[:, 0]))
            if (i + 1) % 12 == 0 or i == N_SEARCHED - 1:
                log(f"  [{arm}] {i + 1}/{N_SEARCHED} states ({(time.time() - t0) / 60:.1f}m)")
        arms[arm] = {"V1": V1, "V3": V3, "LAM": LAM, "lam1_frac": lam1_frac,
                     "eff_rank": eff_rank, "MU": MU, "MUraw": MUraw, "halfcos": halfcos}
        npz[f"{arm}_v1"] = V1
        npz[f"{arm}_eigvals10"] = LAM
        npz[f"{arm}_lam1_frac"] = lam1_frac
        npz[f"{arm}_v1_splithalf_abscos"] = halfcos
        log(f"  [{arm}] lambda1 share of total variance: median {np.median(lam1_frac):.4f}; "
            f"effective rank median {np.median(eff_rank):.2f}; v1 split-half |cos| median "
            f"{np.median(halfcos):.4f}")
        flush()

    # ================================================================= the readings
    held_idx = np.asarray([i for i, s in enumerate(sids)
                           if s == "root.like" or s.startswith("root.like.")], np.int64)
    assert len(held_idx) == 15, f"{len(held_idx)} root.like.* states, expected 15"
    trained_idx = np.asarray([i for i in range(N_SEARCHED) if i not in set(held_idx.tolist())])
    npz["heldout_idx"] = held_idx

    # the state-blind leave-one-out target null
    LOO = np.zeros_like(Ru)
    for i in range(N_SEARCHED):
        LOO[i] = np.delete(Ru, i, axis=0).mean(0)
    LOO = unit(LOO)
    RAND = unit(rng.standard_normal((N_SEARCHED, D_TOWER)))

    for arm in ("profile", "belief"):
        A = arms[arm]
        V1, V3, MU = A["V1"], A["V3"], A["MU"]
        cos_signed = (V1 * Ru).sum(1)
        cos_abs = np.abs(cos_signed)
        cos_rand = np.abs((RAND * Ru).sum(1))
        cos_mu = np.abs((unit(MU) * Ru).sum(1))
        cos_muraw = np.abs((unit(A["MUraw"]) * Ru).sum(1))
        cos_loo = np.abs((LOO * Ru).sum(1))
        # cross-state: v1 at state i against a random OTHER state's target (mirrors C1-final 1a)
        oth = np.array([rng.choice([j for j in range(N_SEARCHED) if j != i])
                        for i in range(N_SEARCHED)])
        cos_other = np.abs((V1 * Ru[oth]).sum(1))

        # -- span capture ---------------------------------------------------------------------
        span3 = np.zeros(N_SEARCHED)
        span3_rand = np.zeros(N_SEARCHED)
        for i in range(N_SEARCHED):
            P = V3[i]                                   # (3, 200), orthonormal by construction
            span3[i] = float((P @ Ru[i]) @ (P @ Ru[i]))
            G = np.linalg.qr(np.random.default_rng(SEED + 31 * i).standard_normal((D_TOWER, 3)))[0]
            span3_rand[i] = float((G.T @ Ru[i]) @ (G.T @ Ru[i]))

        # -- answerability --------------------------------------------------------------------
        cospool = np.abs(V1 @ Qpool_u.T)                 # (72, 1000)
        cosbank = np.abs(V1 @ Qbank_u.T)                 # (72, 19390)
        best_pool = cospool.max(1)
        best_pool_j = cospool.argmax(1)
        best_bank = cosbank.max(1)
        top1_idx = np.asarray([int(meta[i]["top_m_pool_idx"][0]) for i in range(N_SEARCHED)])
        cos_top1_meas = np.abs((V1 * Qpool_u[top1_idx]).sum(1))
        gap = cos_abs - cos_top1_meas
        # instrument references: how much of the TARGET one answerable question already carries
        tgt_top1 = np.abs((Ru * Qpool_u[top1_idx]).sum(1))
        tgt_best_pool = np.abs(Ru @ Qpool_u.T).max(1)

        # -- the selector ---------------------------------------------------------------------
        sp_v1, sp_mu, sp_pop, sp_rand = ([] for _ in range(4))
        nd_v1, nd_mu, nd_pop, nd_rand = ([] for _ in range(4))
        for i, s in enumerate(states):
            asked = {pool_index[q] for q in s["prefix"]}
            cand = np.asarray([q for q in sorted(s["gains"]) if q not in asked], np.int64)
            g = np.asarray([s["gains"][int(q)] for q in cand], np.float64)
            Q = Qpool_u[cand]
            sc = {"v1": np.abs(Q @ V1[i]),
                  "mu": np.abs(Q @ unit(MU[i])),
                  "pop": ans_rate[cand],
                  "rand": np.random.default_rng(SEED + 101 * i).standard_normal(len(cand))}
            for k, acc_s, acc_n in (("v1", sp_v1, nd_v1), ("mu", sp_mu, nd_mu),
                                    ("pop", sp_pop, nd_pop), ("rand", sp_rand, nd_rand)):
                r, _p = spearman(sc[k], g)
                acc_s.append(r)
                acc_n.append(ndcg_at_k(np.argsort(-sc[k]), g, 5))

        def blk(sel, label):
            return {
                "n_states": int(len(sel)),
                "HEADLINE_abs_cos_v1_target": boot_median(cos_abs[sel]),
                "signed_cos_v1_target_declared_convention": boot_median(cos_signed[sel]),
                "null_random_direction": boot_median(cos_rand[sel]),
                "null_cohort_mean_direction_minus_root_mean": boot_median(cos_mu[sel]),
                "null_cohort_mean_direction_raw": boot_median(cos_muraw[sel]),
                "null_leave_one_out_mean_of_other_targets": boot_median(cos_loo[sel]),
                "null_v1_vs_a_random_OTHER_states_target": boot_median(cos_other[sel]),
                "paired_v1_minus_random": boot_paired(cos_abs[sel], cos_rand[sel]),
                "paired_v1_minus_cohort_mean": boot_paired(cos_abs[sel], cos_mu[sel]),
                "paired_v1_minus_leave_one_out": boot_paired(cos_abs[sel], cos_loo[sel]),
                "paired_v1_own_minus_other_state": boot_paired(cos_abs[sel], cos_other[sel]),
                "span3_capture_fraction_of_target_norm2": boot_median(span3[sel]),
                "span3_capture_random_3_directions": boot_median(span3_rand[sel]),
                "paired_span3_minus_random3": boot_paired(span3[sel], span3_rand[sel]),
                "answerability": {
                    "abs_cos_v1_nearest_ANSWERABLE_pool_entry": boot_median(best_pool[sel]),
                    "abs_cos_v1_nearest_entry_in_the_FULL_19390_bank": boot_median(best_bank[sel]),
                    "abs_cos_v1_top1_MEASURED_candidate": boot_median(cos_top1_meas[sel]),
                    "GAP_target_minus_top1_measured": boot_median(gap[sel]),
                    "reference_abs_cos_TARGET_top1_measured": boot_median(tgt_top1[sel]),
                    "reference_abs_cos_TARGET_nearest_pool_entry": boot_median(tgt_best_pool[sel]),
                    "nearest_pool_entry_is_the_top1_measured_candidate": float(
                        np.mean(best_pool_j[sel] == top1_idx[sel]))},
                "selector": {
                    "spearman_vs_measured_gain": {
                        "v1": boot_median(np.asarray(sp_v1)[sel]),
                        "cohort_mean_direction": boot_median(np.asarray(sp_mu)[sel]),
                        "answer_rate_popularity_prior": boot_median(np.asarray(sp_pop)[sel]),
                        "random": boot_median(np.asarray(sp_rand)[sel])},
                    "ndcg_at_5_gain_as_relevance": {
                        "v1": boot_median(np.asarray(nd_v1)[sel]),
                        "cohort_mean_direction": boot_median(np.asarray(nd_mu)[sel]),
                        "answer_rate_popularity_prior": boot_median(np.asarray(nd_pop)[sel]),
                        "random": boot_median(np.asarray(nd_rand)[sel])},
                    "paired_v1_minus_popularity_spearman": boot_paired(
                        np.asarray(sp_v1)[sel], np.asarray(sp_pop)[sel])},
                "spectrum": {"lambda1_share_of_total_variance": boot_median(A["lam1_frac"][sel]),
                             "effective_rank_exp_entropy": boot_median(A["eff_rank"][sel]),
                             "v1_split_half_abs_cos": boot_median(A["halfcos"][sel])},
                "label": label}

        out["readings"][arm] = {
            "all_72_states": blk(np.arange(N_SEARCHED), "all 72 searched node states"),
            "c1_final_TRAINED_57": blk(trained_idx, "the 57 states C1-final was trained on"),
            "c1_final_HELD_OUT_15_root_like": dict(
                blk(held_idx, "the 15 root.like.* states C1-final held out"),
                c1_final_heldout_cos_for_comparison=C1F_HELDOUT_COS,
                comparison_caveat=(
                    "NOT the same space. C1-final's -0.11948 is cos(z_hat, target) in the 129-d "
                    "homogeneous [k(phi), b(phi)] head space; this row is |cos(v1, R_s)| in the "
                    "200-d tower space. The comparable statement is directional only: the "
                    "closed-form construction needs no training, so whatever it scores at trained "
                    "states it scores at held-out states."))}
        npz[f"{arm}_abscos_v1_target"] = cos_abs
        npz[f"{arm}_cos_v1_target_signed"] = cos_signed
        npz[f"{arm}_abscos_rand"] = cos_rand
        npz[f"{arm}_abscos_mu"] = cos_mu
        npz[f"{arm}_abscos_loo"] = cos_loo
        npz[f"{arm}_span3"] = span3
        npz[f"{arm}_span3_rand"] = span3_rand
        npz[f"{arm}_best_pool"] = best_pool
        npz[f"{arm}_cos_top1_meas"] = cos_top1_meas
        npz[f"{arm}_spearman_v1"] = np.asarray(sp_v1)
        npz[f"{arm}_spearman_pop"] = np.asarray(sp_pop)
        npz[f"{arm}_spearman_mu"] = np.asarray(sp_mu)
        npz[f"{arm}_ndcg5_v1"] = np.asarray(nd_v1)
        npz[f"{arm}_ndcg5_pop"] = np.asarray(nd_pop)
        flush()

        a = out["readings"][arm]["all_72_states"]
        log("=" * 100)
        log(f"[{arm}] HEADLINE |cos(v1, target)| median "
            f"{a['HEADLINE_abs_cos_v1_target']['median']:.5f} "
            f"[{a['HEADLINE_abs_cos_v1_target']['lo']:.5f}, "
            f"{a['HEADLINE_abs_cos_v1_target']['hi']:.5f}] "
            f"IQR [{a['HEADLINE_abs_cos_v1_target']['q1']:.5f}, "
            f"{a['HEADLINE_abs_cos_v1_target']['q3']:.5f}]")
        log(f"[{arm}] nulls -- random {a['null_random_direction']['median']:.5f} | cohort mean "
            f"{a['null_cohort_mean_direction_minus_root_mean']['median']:.5f} | LOO other targets "
            f"{a['null_leave_one_out_mean_of_other_targets']['median']:.5f}")
        log(f"[{arm}] span3 capture {a['span3_capture_fraction_of_target_norm2']['median']:.5f} "
            f"vs random-3 {a['span3_capture_random_3_directions']['median']:.5f}")
        log(f"[{arm}] answerability -- nearest pool entry "
            f"{a['answerability']['abs_cos_v1_nearest_ANSWERABLE_pool_entry']['median']:.5f}, "
            f"top-1 measured {a['answerability']['abs_cos_v1_top1_MEASURED_candidate']['median']:.5f}"
            f", GAP {a['answerability']['GAP_target_minus_top1_measured']['median']:+.5f}")
        log(f"[{arm}] selector Spearman v1 "
            f"{a['selector']['spearman_vs_measured_gain']['v1']['median']:+.5f} | popularity "
            f"{a['selector']['spearman_vs_measured_gain']['answer_rate_popularity_prior']['median']:+.5f}"
            f" | cohort mean "
            f"{a['selector']['spearman_vs_measured_gain']['cohort_mean_direction']['median']:+.5f}"
            f" | random "
            f"{a['selector']['spearman_vs_measured_gain']['random']['median']:+.5f}")
        log(f"[{arm}] NDCG@5 v1 {a['selector']['ndcg_at_5_gain_as_relevance']['v1']['median']:.5f} "
            f"| popularity "
            f"{a['selector']['ndcg_at_5_gain_as_relevance']['answer_rate_popularity_prior']['median']:.5f}"
            f" | random {a['selector']['ndcg_at_5_gain_as_relevance']['random']['median']:.5f}")
        h = out["readings"][arm]["c1_final_HELD_OUT_15_root_like"]
        log(f"[{arm}] HELD-OUT 15: |cos| {h['HEADLINE_abs_cos_v1_target']['median']:.5f} "
            f"[{h['HEADLINE_abs_cos_v1_target']['lo']:.5f}, "
            f"{h['HEADLINE_abs_cos_v1_target']['hi']:.5f}] | v1 split-half "
            f"{h['spectrum']['v1_split_half_abs_cos']['median']:.5f}")

    p = out["readings"]["profile"]["all_72_states"]
    out["VERDICT"] = {
        "headline_abs_cos_v1_target_profile_arm": p["HEADLINE_abs_cos_v1_target"]["median"],
        "beats_random_direction": p["paired_v1_minus_random"]["excludes_zero"],
        "beats_cohort_mean_direction": p["paired_v1_minus_cohort_mean"]["excludes_zero"],
        "beats_state_blind_leave_one_out": p["paired_v1_minus_leave_one_out"]["excludes_zero"],
        "selector_beats_popularity": p["selector"]["paired_v1_minus_popularity_spearman"][
            "excludes_zero"],
        "note": "the conclusion is the author's; this block only states which contrasts fired"}
    flush()
    log(f"[done] wrote {OUT_JSON} in {out['minutes']:.1f} minutes")
    return out


def main():
    ap = argparse.ArgumentParser(
        description="Paper C / C6 -- is the measured-best direction the principal disagreement axis?")
    ap.add_argument("--n_users", type=int, default=10000, help="the tree's own train slice")
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--snapshot", default=None)
    ap.add_argument("--concept_ckpt", default=None)
    ap.add_argument("--allow_dirty", action="store_true",
                    help="EMERGENCY ONLY: run importing uncommitted src/instrument (HARD RULE 10)")
    args = ap.parse_args()
    from strategy_channel_suite import CERT_TOWER, CERT_CONC
    args.snapshot = args.snapshot or CERT_TOWER
    args.concept_ckpt = args.concept_ckpt or CERT_CONC
    log("#" * 100)
    log("# PAPER C / C6 -- the closed-form eigen-emitter hypothesis, measured over the 72 searched "
        "node states of coldtree_v2_train10000. TRAIN-SIDE ONLY; nothing is trained; the 10,000 "
        "canonical TEST users are not built, not loaded, not touched.")
    log("#" * 100)
    return run(args)


if __name__ == "__main__":
    main()
