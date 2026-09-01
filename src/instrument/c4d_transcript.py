r"""PAPER C / STAGE C4D -- THE SYNTHESISED-INTERVIEW TRANSCRIPT (the capstone figure's machinery).

Spec: `docs/results/C4D_TRANSCRIPT_SPEC.md` (what the figure shows and what it may NOT claim).
Prerequisites, all IMPORTED and never reimplemented:
  `c1_final.py`   the policy that emits (its 57 TRAINED states are the only states with an emission;
                  `docs/results/C1_FINAL.md`: held-out-state emissions are out of scale and EXCLUDED)
  `c3b_adapter`   the renderer (TowerBridge, the adapter, the SBERT->concept select, the fold)
  `c4c_decoding`  best-of-8 by LM likelihood -- THE DECODING OF RECORD (`docs/results/C4C_DECODING.md`
                  ruling 1); ruling 2 also requires the demo table to carry GREEDY beside it, so it
                  does
  `c4_pilot`      R_s (the state's own measured-ingredient direction) and the fold-back path
  `c4b_noise`     the DECLARED-CIRCULAR answer model, and the rule that every absolute row carrying
                  it is labelled with that phrase wherever it appears
  `tree_route`    the canonical routing of the cold tree, used as a GATE on this file's walk
  `ndcg_tree`     `branch_of` -- the tree's own three-way answer rule, for the walk itself

WHAT THIS PRODUCES
------------------
ONE cold-start interview, turn by turn, for a cohort walking Paper B's cold tree
(`coldtree_v2_train10000`) over the VALIDATION cohort, with three columns per turn:

  1 the SYNTHESISED question   the C1-final policy's emitted off-catalogue direction at that node
                               state, rendered into English by the adapter (best-of-8, greedy beside
                               it), then read back through SBERT -> concepts -> the certified fold;
  2 the CATALOGUE question     what the tree would actually have asked at that state -- the honest
                               comparator, and the visual point of the figure;
  3 the CONSEQUENCE            the cohort's answers under the declared-circular simulator, the
                               cumulative fold, the top-5 recommendations after that turn, and
                               full/tail NDCG@10 through `genv_eval` (the certified contract, the
                               only scorer that exists here).

★ THE HONESTY RULE, AND IT IS THE FIGURE'S MOST IMPORTANT FEATURE
------------------------------------------------------------------
C1-final emits at 57 states and nowhere else. If the walk reaches a node OUTSIDE that set the row
prints the TREE's own question and is marked `[no emission -- outside the fitted set]`. Nothing is
invented, interpolated or borrowed from a neighbouring state. The whole `root.like.*` subtree is
held out in C1-final's primary cell, so a walk that turns `like` at the root falls off the fitted set
immediately -- and that is a row of the figure, not a bug in it.

★ THE ONE PLACE THE ARTIFACTS FORCED A CHOICE: WHICH SPACE THE EMISSION IS RENDERED IN
--------------------------------------------------------------------------------------
The C4 pilot rendered points from the E1 policy's own `k(phi)` space, which is the space the C3B
adapter was TRAINED on (`c3b_pairs.npz`: `K`, `mu`, `scale`, all built from E1's head). C1-FINAL
changed warm start: its policy is Paper B's `m7dj_policy.pt`, so its emissions live in m7dj's
`k(phi)` space. Both are 128-dimensional and both read the SAME frozen `phi` descriptors (item
popularity, decoder rows and concept embeddings are all derived from `train.csv` and are identical
for every cohort -- `run_battery_phaseA.build_real_ctx`), but they are two different learned
read-outs and NOTHING guarantees a shared basis. C1-final banks no policy checkpoint, only the
per-state emitted vectors, so this cannot be sidestepped by re-emitting.

So the space is a DECLARED, MEASURED flag rather than an assumption. Three constructions are built,
all three are rendered, and the snap diagnostic for all three is banked so the choice can be made on
evidence rather than on hope:

  --space direct   (default) the emission's own 128-d vector, mean-removed and scaled by m7dj's OWN
                   bank constants, fed to the adapter as if the two k(phi) spaces coincided. The
                   literal reading of `DS_C3B_V2_QUESTIONS.md`'s demo instruction. Faithful to the
                   emission, and its basis assumption is exactly what the snap diagnostic tests.
  --space lstsq    the BASIS-FREE route: the emission's score profile over the whole 19,390-entry
                   bank (`z_hat . e_q`, the quantity the policy's own argmax reads) is re-expressed
                   in the adapter's space by least squares against E1's bank, standardised, and
                   rescaled to the emission's own relative mean-removed norm. R^2 of that fit is
                   reported; a low R^2 is a REASON NOT TO USE THIS ROW, not a number to quote.
  --space target   the state's m=5 measured-gain BLEND rebuilt directly out of the adapter's own bank
                   rows. No bridge at all, fully in-distribution -- but it renders the state's
                   TARGET, not the policy's emission, and every row it produces is labelled so.
                   (C1-final: emission-to-target cos is 0.970 at trained states, so this is a close
                   proxy -- a proxy is still not the thing.)

THE GATE THAT MAKES THE m7dj REBUILD TRUSTWORTHY: the m=5 blend targets rebuilt here, in the
VALIDATION env, are asserted BIT-IDENTICAL to the 72 rows `c1_final_per_state.npz` banked from the
TRAIN env. If phi were not split-invariant, or the pool/norm/checkpoint were not m7dj's, that assert
fails and the run stops.

TWO MORE DECLARED CHOICES (both recorded in the JSON; neither can move a difference, because both
columns get the identical treatment)
  (i)  BRANCHING is the tree's own behavioural rule (`ndcg_tree.branch_of`), decided BEFORE the
       circular answer model is installed -- so the walk is the tree's real walk, on real answers.
       FOLDING and SCORING then run under the declared-circular model, because an off-catalogue
       synthesised question has no behavioural answer table entry at all and refusal is never
       simulated there; using a different rule for the comparator column would make the two columns
       incomparable, which is precisely what the figure exists to compare. Symmetry wins; the
       inconsistency is declared here and in the spec, never hidden.
  (ii) the fold is CUMULATIVE (the belief updates turn by turn, which is the point of a transcript),
       concepts ascending by id within a turn as `c4b_noise` fixes the ask order, and each turn is
       scored on the cohort standing at THAT node against that same cohort's own m=0 base.

WHAT MAY NOT BE CLAIMED. No value claim of any kind -- the NDCG rows run under a declared-circular
answer model and carry a RELATIVE reading at most (AUTHOR RULING 2026-08-25 stands) - nothing about
states outside C1-final's 57 - nothing about the item channel - that the questions are good because
they read well - that a human would answer them (no human study). VALIDATION ONLY: the 10,000
canonical TEST users are not built, not loaded, not touched. No LLM API.
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)

_NT = "4"
for _i, _a in enumerate(sys.argv):
    if _a == "--threads" and _i + 1 < len(sys.argv):
        _NT = sys.argv[_i + 1]
_NT = str(min(int(_NT), 4))                      # 4 PHYSICAL cores on this box; never exceed
os.environ["OMP_NUM_THREADS"] = _NT
os.environ.setdefault("MKL_NUM_THREADS", _NT)
os.environ.setdefault("OPENBLAS_NUM_THREADS", _NT)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import json
import time
import argparse
import numpy as np
import torch

from policy_net import Policy, build_phi, log, SEED
from c0_geometry import (PAPERC, E1_CKPT, N_BANK, VAL_M0_FULL, VAL_M0_TAIL, sha256,
                         savez_atomic, index_map_assert, bank_vectors)
import c1_pilot as C1
import c1_final as C1F
import c3_poc_extract as POC
import c3b_adapter as C3B
import c4_pilot as C4
import c4b_noise as C4B
import c4c_decoding as C4C
import tree_route

OUT_DIR = os.path.join(PAPERC, "c4d")
OUT_JSON = os.path.join(OUT_DIR, "c4d_transcript.json")
OUT_NPZ = os.path.join(OUT_DIR, "c4d_transcript_per_turn.npz")
OUT_MD = os.path.join(OUT_DIR, "c4d_transcript.md")
OUT_TEX = os.path.join(OUT_DIR, "c4d_transcript.tex")

C1F_JSON = os.path.join(PAPERC, "c1_final.json")
C1F_NPZ = os.path.join(PAPERC, "c1_final_per_state.npz")
CELL = "primary_root_like_subtree"        # the cell whose trained set is the 57 states
EMIT_KEY = f"zbar_{CELL}"
M = 5                                     # the registered blend width, C1-final's own
N_SEARCHED = 72                           # searched node states in coldtree_v2_train10000
N_TRAINED = 57                            # of which C1-final's primary cell trains on 57
TOPK_ASK = C4B.TOPK                       # 4 concepts per rendered question, as C4B fixes it
TOPK_REC = 5                              # the transcript's top-5 recommendation display
CIRC = C4B.CIRC                           # "under a declared-circular answer model"
DEC_PRIMARY = "bestofk_lm"                # C4C ruling 1: the decoding of record
DEC_ARMS = ("bestofk_lm", "greedy")       # C4C ruling 2: greedy is shown beside it
SPACES = ("direct", "lstsq", "target")
BRANCHES = ("like", "dislike", "noanswer")


def write_atomic(path, text):
    with open(path + ".tmp", "w", encoding="utf-8") as fh:
        fh.write(text)
    os.replace(path + ".tmp", path)
    return path


# ============================================================================ the m7dj side
def build_m7dj_bank(E, full_pool, pool, warm_path):
    """m7dj's `[k(phi), b(phi)]` bank, rebuilt inside the VALIDATION env.

    Legitimate because `phi` is split-invariant: the item popularity counts, the head mask, the
    decoder rows and the concept embeddings all come from `train.csv` and the certified snapshot, so
    `build_real_ctx` states in its own docstring that "the ruler does not move when the cohort does".
    That is not taken on trust -- `run()` asserts the blend targets rebuilt from this bank are
    BIT-IDENTICAL to the ones `c1_final` banked from the train env.
    """
    blob = torch.load(warm_path, map_location="cpu")
    assert blob["arch"] == "equivariant" and blob["state"] == "both"
    assert len(blob["ext"]) == C1F.N_EXT, f"m7dj's phi bank is {len(blob['ext'])}, not {C1F.N_EXT}"
    assert int(blob["n_pool"]) == 1000 and int(blob["d_phi"]) == 205 and int(blob["d_z"]) == 200
    assert pool == [(int(a), int(b)) for a, b in blob["pool"]], "the env's pool is not m7dj's pool"
    ext = [(int(a), int(b)) for a, b in blob["ext"]]
    norm = (np.asarray(blob["norm"][0], np.float32), np.asarray(blob["norm"][1], np.float32))
    phi_ext, _ = build_phi(E, ext, norm=norm)
    assert phi_ext.shape == (C1F.N_EXT, 205)
    phi_full, _ = build_phi(E, full_pool, norm=norm)
    pol = Policy(200, 205, len(pool), arch="equivariant", state="both", use_pos=True)
    pol.load_state_dict(blob["sd"]); pol.eval()
    assert sum(p.numel() for p in pol.parameters()) == 526542
    Ebank, Kbank = bank_vectors(pol, phi_full)
    Epool, Kpool = bank_vectors(pol, phi_ext[:len(pool)])
    Kb = Kbank.astype(np.float64)
    muK = Kb.mean(0)
    return {"blob": blob, "pol": pol, "Ebank": Ebank, "Kbank": Kbank,
            "Epool": Epool, "Kpool": Kpool, "sha256": sha256(warm_path), "muK": muK,
            "scaleK": float(np.linalg.norm(Kb - muK, axis=1).mean())}


# ============================================================================ the three spaces
def space_points(space, idx, Zemit, m7, Kbank_E1, mu_E1, scale_E1, meta, pool2full):
    """One declared construction of the adapter-space point for each walked state.

    Returns `(P, mu, scale, diag)`. `P` is a RAW k(phi)-space point per state and `(mu, scale)` are
    the constants it must be standardised by; the caller hands all three to `C3B.input_vectors`, so
    the standardisation happens exactly once and in the adapter's own function.
    """
    muK7, sc7 = m7["muK"], m7["scaleK"]
    if space == "direct":
        return (Zemit[idx, :-1].astype(np.float64), muK7, sc7,
                {"construction": "the emission's own k(phi) vector, standardised by m7dj's OWN bank "
                                 "mean and global scale (so its LENGTH is in the adapter's units "
                                 "even though its BASIS is m7dj's)",
                 "mu_norm": float(np.linalg.norm(muK7)), "scale": sc7,
                 "renders": "the POLICY'S EMISSION"})
    if space == "target":
        P = np.zeros((len(idx), Kbank_E1.shape[1]), np.float64)
        for j, i in enumerate(idx):
            q = [int(x) for x in meta[int(i)]["top_m_pool_idx"]]
            w = np.asarray(meta[int(i)]["weights"], np.float64)
            b = np.asarray([int(pool2full[x]) for x in q], np.int64)
            P[j] = (w[:, None] * Kbank_E1[b]).sum(0)
        return P, mu_E1, scale_E1, {
            "construction": "the state's m=5 measured-gain blend rebuilt out of the ADAPTER'S OWN "
                            "bank rows -- no cross-space bridge exists in this row",
            "renders": "the state's BLEND TARGET, not the policy's emission (labelled in every row)"}
    # ---- lstsq: the basis-free route ------------------------------------------------------------
    S = Zemit[idx].astype(np.float64) @ m7["Ebank"].astype(np.float64).T      # (k, 19390) scores
    Kc = Kbank_E1 - mu_E1                                                     # centred E1 bank
    G = Kc.T @ Kc
    G = G + np.eye(G.shape[0]) * (1e-8 * float(np.trace(G)) / G.shape[0])     # ridge, declared
    P = np.zeros((len(idx), Kbank_E1.shape[1]), np.float64)
    r2 = np.zeros(len(idx))
    for j in range(len(idx)):
        s = S[j] - S[j].mean()
        s = s / max(np.linalg.norm(s), 1e-30)
        p = np.linalg.solve(G, Kc.T @ s)
        res = s - Kc @ p
        r2[j] = 1.0 - float(res @ res) / max(float(s @ s), 1e-30)
        # rescale to the emission's OWN relative mean-removed length, so the adapter sees a vector of
        # the length its training distribution has (the C1-final scale term's whole purpose)
        want = float(np.linalg.norm(Zemit[idx[j], :-1].astype(np.float64) - muK7)) / sc7 * scale_E1
        P[j] = mu_E1 + p * (want / max(float(np.linalg.norm(p)), 1e-30))
    return P, mu_E1, scale_E1, {
        "construction": ("the emission's SCORE PROFILE over the 19,390-entry bank (z_hat . e_q, the "
                         "quantity the policy's own argmax reads) re-expressed in the adapter's "
                         "space by ridge-stabilised least squares, standardised, then rescaled to "
                         "the emission's own relative mean-removed norm"),
        "r2_per_state": [float(x) for x in r2], "r2_median": float(np.median(r2)),
        "r2_note": "a low R^2 is a REASON NOT TO USE THIS ROW, not a number to quote",
        "renders": "the POLICY'S EMISSION, re-expressed"}


# ============================================================================ the walk
def walk(E, nodes, rows, mode, max_turns, emitting):
    """Descend ONE path of the cold tree over `rows`, under the tree's own behavioural answer rule.

    Called BEFORE the declared-circular answer model is installed, so the branching is the tree's
    real branching on real answers. Returns the per-turn record; the question sequence it produces is
    GATED against `tree_route.route`'s own output for the users who end up on this path.
    """
    from ndcg_tree import branch_of
    explicit = None
    if mode not in ("majority", "emitting"):
        parts = mode.split(".")
        assert parts[0] == "root", "--path must be 'majority', 'emitting' or a 'root.x.y' path"
        explicit = parts[1:]
        assert all(p in BRANCHES for p in explicit), f"bad branch name in --path {mode!r}"
    nid, cohort, turns = "root", np.asarray(rows, np.int64), []
    stop = None
    while len(turns) < max_turns:
        nd = nodes.get(nid)
        if nd is None:
            stop = f"the walk left the fitted tree at {nid}"
            break
        if not nd.get("split"):
            stop = f"{nid} is a leaf -- the tree stops splitting here"
            break
        q = (int(nd["splitter"][0]), int(nd["splitter"][1]))
        lab, ans_rate, n_zero = branch_of(E, cohort, q)
        kids = {b: cohort[lab == b] for b in BRANCHES}
        avail = [b for b in BRANCHES if len(kids[b]) and f"{nid}.{b}" in nodes]
        t = len(turns)
        if explicit is not None:
            if t >= len(explicit):
                pick = None
                stop = f"--path {mode!r} is exhausted at depth {t}"
            else:
                pick = explicit[t]
                assert f"{nid}.{pick}" in nodes, \
                    f"--path asks for {nid}.{pick}, which this tree has no node for"
                assert len(kids[pick]), f"--path asks for {nid}.{pick}, which no cohort user reaches"
        elif not avail:
            pick, stop = None, f"{nid} has no fitted child this cohort populates"
        elif mode == "emitting":
            em = [b for b in avail if f"{nid}.{b}" in emitting]
            pick = max(em or avail, key=lambda b: len(kids[b]))
        else:
            pick = max(avail, key=lambda b: len(kids[b]))
        turns.append({"turn": t + 1, "sid": nid, "depth": int(nd["depth"]),
                      "rows": cohort.copy(), "splitter": q,
                      "splitter_gain": float(nd.get("splitter_gain", float("nan"))),
                      "tree_n_users_at_fit": int(nd.get("n_users", -1)),
                      "behavioural_answer_rate": float(ans_rate),
                      "n_zero_value_answers": int(n_zero),
                      "branch_counts": {b: int(len(kids[b])) for b in BRANCHES},
                      "branch_taken": pick,
                      "emits": nid in emitting})
        if pick is None:
            break
        nid = f"{nid}.{pick}"
        cohort = kids[pick]
    else:
        stop = f"--max_turns {max_turns} reached"
    return turns, (stop or "walk ended")


def route_gate(E, nodes, turns, total_L):
    """`tree_route.route`'s own interview for the users on the walked path must AGREE, question by
    question, with the splitters this walk asked. A MATCH RATE is asserted, never non-emptiness
    (memory `cross-population-index-match-is-silent`)."""
    last = turns[-1]
    rows = [int(r) for r in last["rows"]]
    seqs, diag = tree_route.route(E, rows, nodes, total_L)
    want = [t["splitter"] for t in turns]            # every walked node's own splitter, in ask order
    n_ok = 0
    for s in seqs:
        if [tuple(q) for q in s[:len(want)]] == want:
            n_ok += 1
    rate = n_ok / max(len(rows), 1)
    assert rate == 1.0, (
        f"tree_route's own interview disagrees with this walk for {len(rows) - n_ok} of "
        f"{len(rows)} users on the walked path (match rate {rate:.4f}) -- the walk is not the "
        f"tree's own descent, STOP")
    return {"n_users_on_the_walked_path": len(rows), "match_rate": rate,
            "n_questions_compared": len(want), "route_n_fallback": int(diag["n_fallback"])}


# ============================================================================ recommendations
def mean_scores(E, seq, rows, batch=500):
    """The cohort's MEAN item score after folding `seq` -- the display object behind the top-5.

    DISPLAY ONLY, and labelled so everywhere: it is the cohort mean of the decoder's own score
    (`z @ Wd.T + bd`, the identical expression `genv_eval` scores), NOT masked by the per-user
    policy-B pool, because a pool is per user and a cohort-level list is not. Every NDCG number in
    this file comes from `genv_eval` under the certified contract and none of them come from here.
    """
    from strategy_channel_suite import genv_latent
    ctx = E["ctx"]
    z = genv_latent(E, seq, rows)
    acc = np.zeros(int(ctx.ni), np.float64)
    with torch.no_grad():
        for st in range(0, len(rows), batch):
            S = (z[st:st + batch] @ ctx.Wd.T + ctx.bd).numpy().astype(np.float64)
            acc += S.sum(0)
    return acc / max(len(rows), 1)


def top_items(m, names, k=TOPK_REC, prev=None):
    o = np.argsort(-m)[:k]
    out = [{"rank": int(r + 1), "item": int(i), "title": POC.phrase(names, len(m), int(i)),
            "mean_score": float(m[i])} for r, i in enumerate(o)]
    if prev is not None:
        d = m - prev
        for r, i in enumerate(np.argsort(-d)[:k]):
            out.append({"riser_rank": int(r + 1), "item": int(i),
                        "title": POC.phrase(names, len(m), int(i)),
                        "mean_score_gain": float(d[i])})
    return out


# ============================================================================ the run
def run(args):
    from ndcg_tree import build_pool, leak_assert
    from strategy_channel_suite import genv_mean, genv_eval, contract_selftest
    t00 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    GB = C4C.import_scoped_git_block(args)                              # HARD RULE 10, first
    log(f"[hardrule10] rev {GB['rev'][:8]}; imported src/instrument modules "
        f"{GB['n_modules_imported_from_src_instrument']}; dirty AND imported "
        f"{len(GB['dirty_src_instrument_files_imported_by_this_run'])}")

    # ---- the renderer's bank: `phrases` (C3B) or `questions` (c3b_v2) ----------------------------
    mode = {"phrase": "phrases", "phrases": "phrases", "v2": "questions",
            "questions": "questions"}[args.adapter]
    C3B.set_targets(mode)
    ad_path = os.path.join(C3B.OUT_DIR, "c3b_true_best.pt")
    assert os.path.exists(ad_path), (
        f"--adapter {args.adapter} wants {ad_path}, which does not exist yet. The c3b_v2 arm writes "
        f"`c3b_true_best.pt` only when its training run finishes selecting a best epoch.")
    max_new = C3B.max_new_tokens()
    log(f"[adapter] targets={mode} dir={os.path.relpath(C3B.OUT_DIR, _ROOT)} "
        f"max_new_tokens={max_new}")

    # ---- the bridge: env + tower targets + SBERT (validation, the C4 measurement space) ----------
    br = C3B.TowerBridge(args)
    E, ni, C = br.E, br.ni, br.C
    ck_sha = sha256(args.ckpt)
    assert ck_sha == C1.C0_CKPT_SHA == br.ckpt_sha

    # HARD RULE 5, before anything: this is the certified ruler, not a lookalike env
    f0, t0_ = genv_mean(E, [], E["rows"])
    assert round(f0, 8) == round(VAL_M0_FULL, 8), f"canonical snap FAILED (full): {f0!r}"
    assert round(t0_, 8) == round(VAL_M0_TAIL, 8), f"canonical snap FAILED (tail): {t0_!r}"
    log(f"[gate] validation m=0 intercept full={f0!r} tail={t0_!r} -- both canonical")
    contract_selftest(E, E["rows"])
    leak_assert(E, E["rows"], "paperC/c4d")

    pool, _it, _cc = build_pool(E, 500, 500)
    pool = [(int(a), int(b)) for a, b in pool]
    pool_index = {q: i for i, q in enumerate(pool)}
    assert ni + C == N_BANK, f"bank composition {ni}+{C} != {N_BANK}"
    pool2full = np.asarray([(i if ch == 0 else ni + i) for ch, i in pool], np.int64)
    full_pool = [(0, i) for i in range(ni)] + [(1, c) for c in range(C)]
    assert len(full_pool) == N_BANK

    # ---- the adapter's own space constants: READ from the C3B pairs, never recomputed ------------
    man = C3B.load_manifest()
    zp = np.load(C3B.PAIRS_NPZ)
    mu_E1 = np.asarray(zp["mu"], np.float64)
    scale_E1 = float(zp["scale"])
    Kbank_E1 = np.asarray(zp["K"], np.float64)
    assert Kbank_E1.shape == (N_BANK, 128)
    d_mu = float(np.abs(Kbank_E1.mean(0) - mu_E1).max())
    assert d_mu < 1e-9, f"the banked mean-removal constant is not the bank mean (max|diff| {d_mu:g})"
    assert abs(float(man["space"]["global_scale_mean_centred_norm"]) - scale_E1) < 1e-12
    assert man["checkpoint"]["sha256"] == ck_sha, "the C3B pairs were built on another checkpoint"
    # the v2 (question-target) run rebuilds its own pairs file; only the TARGETS moved, so its input
    # space must be byte-for-byte the C3B one. Asserted rather than assumed -- if it ever is not, the
    # `target` space construction below would be mixing two banks.
    v1_pairs = os.path.join(PAPERC, "c3b", "c3b_pairs.npz")
    space_same = None
    if os.path.abspath(C3B.PAIRS_NPZ) != os.path.abspath(v1_pairs) and os.path.exists(v1_pairs):
        z1 = np.load(v1_pairs)
        space_same = {"mu_max_abs_diff": float(np.abs(np.asarray(z1["mu"], np.float64) - mu_E1).max()),
                      "scale_abs_diff": abs(float(z1["scale"]) - scale_E1),
                      "K_max_abs_diff": float(np.abs(np.asarray(z1["K"], np.float64)
                                                     - Kbank_E1).max())}
        assert max(space_same.values()) == 0.0, (
            f"the selected adapter's input space is not the C3B one: {space_same}")
        log("[space] the v2 pairs file carries the IDENTICAL input space as C3B's (mu, scale, K all "
            "bit-identical); only the TARGETS moved")
    blob1 = torch.load(args.ckpt, map_location="cpu")
    assert pool == [(int(a), int(b)) for a, b in blob1["pool"]], "the env's pool is not E1's pool"
    phi1, _ = build_phi(E, pool, norm=(blob1["norm_mu"], blob1["norm_sd"]))
    pol1 = Policy(E["ctx"].d, phi1.shape[1], N_BANK, arch=blob1["arch"], state=blob1["state"],
                  use_pos=blob1["use_pos"])
    pol1.load_state_dict(blob1["model"]); pol1.eval()
    _Ep1, Kp1 = bank_vectors(pol1, phi1)
    map_err = index_map_assert(Kbank_E1, pool2full, Kp1.astype(np.float64))
    log(f"[space/E1] mean-removal ||mu||={np.linalg.norm(mu_E1):.4f}, global scale {scale_E1:.4f}; "
        f"pool2full exact (max|diff| {map_err:g})")

    # ---- the m7dj side, rebuilt in this env, and GATED on C1-final's banked targets ---------------
    m7 = build_m7dj_bank(E, full_pool, pool, C1F.WARM)
    tree = json.load(open(C1F.TREE_JSON, encoding="utf-8"))
    assert tree["split"] == "train" and int(tree["total_L"]) == C1F.TOTAL_L
    assert tree["test_touched"] is False
    nodes = tree["nodes"]
    states, gain_chk = C1F.load_sweeps(nodes, pool_index)
    assert len(states) == N_SEARCHED
    Tt, TKt, meta = C1.build_targets(states, m7["Epool"], m7["Kpool"], pool_index, M)
    zf = np.load(C1F_NPZ, allow_pickle=True)
    sids = [str(x) for x in zf["state_sid"]]
    assert sids == [s["sid"] for s in states], "the sweep states are not c1_final's banked states"
    d_tgt = float(np.abs(Tt - np.asarray(zf[f"target_m{M}"], np.float32)).max())
    assert d_tgt == 0.0, (
        f"★ THE m7dj REBUILD GATE FAILED: the m={M} blend targets rebuilt in the VALIDATION env are "
        f"not the ones c1_final banked from the TRAIN env (max|diff| {d_tgt:g}). Either phi is not "
        f"split-invariant here, or the pool / norm / checkpoint is not m7dj's. STOP.")
    log(f"[gate] the m7dj bank rebuilt in the validation env reproduces c1_final's {N_SEARCHED} "
        f"blend targets BIT-IDENTICALLY (max|diff| {d_tgt:g})")

    # ---- C1-final's emissions and its 57 trained states ------------------------------------------
    with open(C1F_JSON, encoding="utf-8") as fh:
        c1f = json.load(fh)
    assert c1f["analysis"] == "paperC_C1_final" and c1f["test_touched"] is False
    cell = c1f["cells"][CELL]
    assert int(cell["n_trained_states"]) == N_TRAINED and int(cell["n_held_out_states"]) == 15
    g3 = cell["reading3_no_collapse_GATE"]["PASS"]
    g4 = cell["reading4_renderability_GATE"]["PASS"]
    assert g3 and g4, "C1-final's blocking gates did not pass in this artifact -- nothing to render"
    held = set(cell["held_out_sids"])
    emitting = [s for s in sids if s not in held]
    assert len(emitting) == N_TRAINED
    emitting_set = set(emitting)
    Zemit = np.asarray(zf[EMIT_KEY], np.float64)
    assert Zemit.shape == (N_SEARCHED, 129)
    assert float(np.abs(Zemit[:, -1] - 1.0).max()) < 1e-4, \
        "the banked emission's homogeneous coordinate is not pinned to 1"
    cos_own = np.asarray(zf[f"cos_own_{CELL}"], np.float64)
    tr_idx = np.asarray([i for i, s in enumerate(sids) if s in emitting_set], np.int64)
    med_npz = float(np.median(cos_own[tr_idx]))
    med_json = float(cell["reading2_fit_trained_states"]["cos_own_target"]["median"])
    assert abs(med_npz - med_json) < 1e-9, (
        f"the npz's own trained-state cosines ({med_npz:.7f}) disagree with c1_final.json's reported "
        f"median ({med_json:.7f}) -- the emission array and the reported run are not one object")
    log(f"[gate] emissions traced to c1_final rev {c1f['git']['rev'][:8]}: {N_TRAINED} trained "
        f"states, median cos to own target {med_npz:.5f}, norm ratio "
        f"{float(np.median(np.asarray(zf[f'norm_ratio_{CELL}'])[tr_idx])):.4f}")

    # ---- R_s, the state's own measured-ingredient direction (IMPORTED from the pilot) -------------
    R, r_disc = C4.reference_directions(br, meta, pool, pool2full, n_states=N_SEARCHED)
    R_unit = R / np.linalg.norm(R, axis=1, keepdims=True)
    D_unit = br.d_c / np.linalg.norm(br.d_c, axis=1, keepdims=True)
    cos_tower = R_unit @ D_unit.T                             # (72, 1031)

    # ============================================================ THE WALK (behavioural branching)
    rows_all = [int(r) for r in E["rows"]]
    if args.n_users:
        assert args.n_users <= len(rows_all), f"--n_users {args.n_users} > cohort {len(rows_all)}"
        rows_all = rows_all[:args.n_users]
    log(f"[cohort] {len(rows_all):,} validation users "
        f"({'the whole cohort' if not args.n_users else 'a DECLARED leading slice'})")
    turns, stop_reason = walk(E, nodes, rows_all, args.path, args.max_turns, emitting_set)
    assert turns, "the walk produced no turn at all"
    rg = route_gate(E, nodes, turns, C1F.TOTAL_L)
    log(f"[gate] tree_route's own interview matches this walk for {rg['match_rate']:.4f} of the "
        f"{rg['n_users_on_the_walked_path']:,} users on the walked path")
    for t in turns:
        t["state_idx"] = sids.index(t["sid"]) if t["sid"] in sids else -1
    emit_turns = [t for t in turns if t["emits"]]
    log(f"[walk] {len(turns)} turns, {len(emit_turns)} of them at a C1-final EMITTING state; "
        f"stop: {stop_reason}")
    for t in turns:
        bi = int(pool2full[pool_index[t["splitter"]]])
        t["catalogue_question"] = POC.question_text(br.names, ni, bi)
        t["catalogue_phrase"] = POC.phrase(br.names, ni, bi)
        t["catalogue_channel"] = "concept" if t["splitter"][0] == 1 else "item"
        t["catalogue_bank_idx"] = bi

    # ============================================================ RENDER (all three spaces)
    C3B.ensure_lm()
    tok, lm = C3B.load_lm()
    torch.set_num_threads(int(args.threads))
    d_in = int(zp["X"].shape[1])
    ad, bl = C3B.load_adapter(ad_path, d_in)
    assert bl["split_hashes"] == man["split"]["hashes"], \
        "the adapter was trained under a DIFFERENT split than this manifest's"
    log(f"[adapter] A ep{bl['epoch']} val CE {bl['val_ce']:.5f}; split hashes match")
    ch_concept = int([c for c, nm in C4.CHANNELS if nm == args.channel + "_hot"][0])

    ei = np.asarray([t["state_idx"] for t in emit_turns], np.int64)
    n_em = len(ei)
    assert args.gen_batch >= max(n_em, 1), (
        "--gen_batch must cover every emitting turn in ONE batch, so the sampled candidates of "
        "best-of-8 are reproducible from the seed alone (the C4C reproducibility rule)")
    texts, gdiag, sp_diag, points = {}, {}, {}, {}
    if n_em:
        for sp in SPACES:
            P, mu_s, sc_s, dg = space_points(sp, ei, Zemit, m7, Kbank_E1, mu_E1, scale_E1,
                                             meta, pool2full)
            Xin = C3B.input_vectors(P, mu_s, sc_s, np.full(n_em, ch_concept, np.int64))
            assert Xin.shape == (n_em, d_in)
            points[sp] = (P, mu_s, sc_s)
            dg["adapter_input_abs_mean"] = float(np.abs(Xin[:, :-2]).mean())
            dg["adapter_input_mean_removed_norm_median"] = float(
                np.median(np.linalg.norm(Xin[:, :-2], axis=1)))
            for a in DEC_ARMS:
                texts[(sp, a)], gdiag[(sp, a)] = C4C.generate_arm(lm, tok, ad, Xin, a,
                                                                  args.gen_batch, max_new=max_new)
            dg["n_distinct_bestofk"] = int(len(set(texts[(sp, DEC_PRIMARY)])))
            dg["n_empty_generations_declared_non_encodable"] = int(
                sum(1 for x in texts[(sp, DEC_PRIMARY)] if not str(x).strip()))
            sp_diag[sp] = dg
            log(f"  [gen] space={sp:<7} {dg['n_distinct_bestofk']}/{n_em} distinct under "
                f"{DEC_PRIMARY} ({(time.time() - t00) / 60:.1f}m)")

    # ---- the SPACE DIAGNOSTIC: does a fed point still point where the emission pointed? ----------
    # snap_m7  = the bank entry the EMISSION is nearest to, mean-removed, in m7dj's own k-space
    # snap_fed = the bank entry the FED point is nearest to, mean-removed, in the ADAPTER's k-space
    # Both are the same construction in two spaces, so the agreement is apples to apples. It is
    # REPORTED, never asserted: a zero agreement is a finding about the two heads, not a harness
    # failure -- but it is the number that decides whether a `direct` render means anything.
    space_probe = {}
    if n_em:
        Kc_E1u = Kbank_E1 - mu_E1
        Kc_E1u = Kc_E1u / np.linalg.norm(Kc_E1u, axis=1, keepdims=True)
        Kc_7u = m7["Kbank"].astype(np.float64) - m7["muK"]
        Kc_7u = Kc_7u / np.linalg.norm(Kc_7u, axis=1, keepdims=True)
        Z7 = Zemit[ei, :-1] - m7["muK"]
        snap_m7 = np.argmax((Z7 / np.linalg.norm(Z7, axis=1, keepdims=True)) @ Kc_7u.T, axis=1)
        snap_banked = np.asarray(zf[f"emit_snap_{CELL}"], np.int64)[ei]
        # The top-1 arm is EXACT-INDEX equality over a 19,390-entry bank and is therefore brutally
        # strict: two constructions can point into the same small neighbourhood and still never name
        # the identical row. So a NEIGHBOURHOOD arm is banked beside it -- how many of the emission's
        # own five nearest bank entries (in m7dj's space) the fed point also names among its five
        # (in the adapter's space). Chance is 5/19390 = 0.00026 per slot, so anything above ~0 is
        # signal. Neither arm is asserted; both are reported.
        TOPN = 5
        em_top = np.argsort(-((Z7 / np.linalg.norm(Z7, axis=1, keepdims=True)) @ Kc_7u.T),
                            axis=1)[:, :TOPN]
        for sp in SPACES:
            P, mu_s, _sc = points[sp]
            Pc = P - mu_s
            Pu = Pc / np.linalg.norm(Pc, axis=1, keepdims=True)
            Sfed = Pu @ Kc_E1u.T
            snap_fed = np.argmax(Sfed, axis=1)
            fed_top = np.argsort(-Sfed, axis=1)[:, :TOPN]
            ov = np.asarray([len(set(fed_top[j].tolist()) & set(em_top[j].tolist()))
                             for j in range(n_em)], np.int64)
            space_probe[sp] = {
                "n_states": int(n_em),
                "top5_neighbourhood_overlap_mean": float(ov.mean()),
                "top5_neighbourhood_overlap_per_state": [int(x) for x in ov],
                "top5_neighbourhood_overlap_chance": float(TOPN * TOPN / N_BANK),
                "fed_point_top5_bank_phrases":
                    [[POC.phrase(br.names, ni, int(x)) for x in r] for r in fed_top],
                "emission_top5_bank_phrases_in_m7dj_space":
                    [[POC.phrase(br.names, ni, int(x)) for x in r] for r in em_top],
                "snap_agreement_with_the_emission_in_its_own_space":
                    float((snap_fed == snap_m7).mean()),
                "snap_agreement_with_c1_final_banked_emit_snap":
                    float((snap_fed == snap_banked).mean()),
                "fed_point_nearest_bank_phrase":
                    [POC.phrase(br.names, ni, int(x)) for x in snap_fed],
                "emission_nearest_bank_phrase_in_m7dj_space":
                    [POC.phrase(br.names, ni, int(x)) for x in snap_m7],
                "note": ("agreement near zero means the two policies' k(phi) read-outs do NOT share "
                         "a basis, so a `direct` render is a render of a DIFFERENT direction; that "
                         "is a finding to report, not a number to work around. The banked "
                         "`emit_snap` arm uses c1_final's RAW homogeneous argmax with the state's "
                         "prefix excluded, so it can differ from the mean-removed arm by "
                         "construction and is the secondary of the two.")}
            log(f"  [space probe] {sp:<7} nearest-bank agreement with the emission in its own "
                f"space {space_probe[sp]['snap_agreement_with_the_emission_in_its_own_space']:.3f} "
                f"(vs banked emit_snap "
                f"{space_probe[sp]['snap_agreement_with_c1_final_banked_emit_snap']:.3f}); "
                f"top-5 neighbourhood overlap "
                f"{space_probe[sp]['top5_neighbourhood_overlap_mean']:.2f}/5 "
                f"(chance {space_probe[sp]['top5_neighbourhood_overlap_chance']:.5f})")

    # ============================================================ THE ANSWER MODEL (declared circular)
    sgn, _U, ans_diag = C4B.geometric_answers(br, rows_all)
    ins = C4B.install_answer_model(E, rows_all, sgn)
    rowpos = {int(r): j for j, r in enumerate(rows_all)}
    log(f"[answers] {CIRC} installed over {len(rows_all):,} user-rows "
        f"({ans_diag['frac_positive'] * 100:.2f}% positive); replaces a behavioural table at answer "
        f"rate {ins['behavioural_answer_rate_replaced']:.4f}")

    # ============================================================ THE TRANSCRIPT
    # C4_PILOT.md ruling 1: an EMPTY generation is not a rendering -- declared non-encodable, never
    # SBERT-selected, never folded, and counted rather than silently dropped.
    sel_all = {}
    for sp in SPACES:
        for a in DEC_ARMS:
            if (sp, a) in texts:
                tx = texts[(sp, a)]
                empt = [not str(x).strip() for x in tx]
                s = br.select([("" if e else str(x)) for x, e in zip(tx, empt)])
                sel_all[(sp, a)] = [[] if e else q for q, e in zip(s, empt)]
    U_prim = br.tower_vectors(sel_all[(args.space, DEC_PRIMARY)]) if n_em else None
    K7u = m7["Kbank"].astype(np.float64) - m7["muK"]
    K7u = K7u / np.linalg.norm(K7u, axis=1, keepdims=True)
    emit_turn_ids = [x["turn"] for x in emit_turns]

    syn_seq, tree_seq, rowsheet = [], [], []
    npz_out = {"sid": np.asarray([t["sid"] for t in turns], object),
               "emits": np.asarray([t["emits"] for t in turns], bool),
               "state_idx": np.asarray([t["state_idx"] for t in turns], np.int64)}
    for sp in SPACES:
        if sp in points:
            npz_out[f"point_{sp}"] = points[sp][0].astype(np.float32)
        for a in DEC_ARMS:
            if (sp, a) in texts:
                npz_out[f"text_{sp}_{a}"] = np.asarray(texts[(sp, a)], object)
    out = {
        "analysis": "paperC_C4D_transcript",
        "produced_by": "src/instrument/c4d_transcript.py",
        "spec": "docs/results/C4D_TRANSCRIPT_SPEC.md",
        "citable": True, "test_touched": False, "split": "validation",
        "what_this_is": ("ONE cold-start interview walked down Paper B's cold tree, in which every "
                         "question at a C1-final EMITTING state is the policy's own off-catalogue "
                         "direction rendered into English, shown beside the catalogue question the "
                         "tree would have asked at that state"),
        "figure_rule": ("a state outside C1-final's 57 trained states gets NO synthesised question: "
                        "the row prints the tree's own question and is marked "
                        "'[no emission -- outside the fitted set]'"),
        "n_turns": len(turns), "n_turns_emitting": len(emit_turns),
        "walk": {"path_mode": args.path, "stop_reason": stop_reason,
                 "cohort_n_users": len(rows_all),
                 "cohort_is_whole_validation_split": not bool(args.n_users),
                 "branching": ("ndcg_tree.branch_of -- the tree's OWN behavioural three-way rule, "
                               "decided BEFORE the circular answer model is installed"),
                 "route_gate": rg},
        "emission_source": {
            "file": os.path.relpath(C1F_NPZ, _ROOT), "key": EMIT_KEY, "sha256": sha256(C1F_NPZ),
            "cell": CELL, "c1_final_git_rev": c1f["git"]["rev"], "n_trained_states": N_TRAINED,
            "trained_sids": emitting, "held_out_sids": sorted(held),
            "per_state_vector": ("the COHORT MEAN of z_hat at that state. C1-final measures "
                                 "within-state per-user agreement at 0.99991, so the emitted point "
                                 "at a state is one object, not an average over disagreeing users. "
                                 "C1-final banks no policy checkpoint, so re-emitting is not "
                                 "possible and this is the only available object."),
            "median_cos_to_own_target_trained": med_npz},
        "space": {"chosen": args.space, "all_built": list(SPACES),
                  "why_this_is_a_flag": (
                      "C1-final's policy is Paper B's m7dj warm start, so its emissions live in "
                      "m7dj's k(phi) space; the C3B adapter was trained on E1's k(phi) space. Both "
                      "are 128-d over the SAME frozen phi, but they are two learned read-outs and "
                      "nothing guarantees a shared basis. See `probe`."),
                  "constructions": sp_diag, "probe": space_probe,
                  "m7dj_rebuild_gate": {"blend_targets_max_abs_diff_vs_c1_final": d_tgt,
                                        "sweep_argmax_equals_splitter_max_abs_gain_err": gain_chk}},
        "renderer": {"adapter": args.adapter, "targets_mode": mode,
                     "path": os.path.relpath(ad_path, _ROOT),
                     "epoch": bl["epoch"], "val_ce": bl["val_ce"],
                     "max_new_tokens": max_new, "channel_hot": args.channel + "_hot",
                     "decoding_primary": DEC_PRIMARY,
                     "decoding_spec": {a: C4C.DEC_SPEC[a] for a in DEC_ARMS},
                     "decoding_of_record": ("C4C_DECODING.md ruling 1: best-of-8 by LENGTH-"
                                            "NORMALISED LM likelihood, a pick rule blind to every "
                                            "evaluation quantity. Ruling 2: greedy is shown beside "
                                            "it, because the variance is real."),
                     "gen_batch": int(args.gen_batch),
                     "sbert": {"model": br.RE.SBERT_MODEL, "pooling": br.RE.SBERT_MODE,
                               "topn": br.RE.SBERT_TOPN, "mincos": br.RE.SBERT_MINCOS}},
        "answer_model": dict(ans_diag, **ins, label=CIRC),
        "declared_choices": {
            "branch_behavioural_fold_circular": (
                "BRANCHING uses the tree's behavioural answers (so the walk is the tree's real "
                "walk); FOLDING and SCORING use the declared-circular model, because a synthesised "
                "off-catalogue question has no behavioural answer table entry and refusal is never "
                "simulated. Both columns get the identical treatment, so the choice cannot move the "
                "comparison; it is declared here, in the spec and in the caption."),
            "cumulative_fold": ("the belief updates turn by turn: turn t folds everything asked up "
                                "to t, concepts ascending by id within a turn (c4b_noise's ask "
                                "order). Each turn is scored on the cohort standing at THAT node, "
                                "against that same cohort's own m=0 base."),
            "top5_display": ("the top-5 lists are the cohort MEAN of the decoder score, unmasked -- "
                             "a DISPLAY object. Every NDCG number comes from genv_eval under the "
                             "certified contract and none comes from the display path."),
            "cohort": ("the whole 10,000-user validation cohort unless --n_users names a leading "
                       "slice, which is then declared here and in the figure caption")},
        "reference": {"definition": ("R_s = sum_j w_j * tower(q_j) over the state's banked m=5 "
                                     "ingredients (c4_pilot.reference_directions, IMPORTED)"),
                      "disclosure": r_disc},
        "canonical_snap": {"val_m0_full": f0, "val_m0_tail": t0_},
        "checkpoint": {"path": os.path.relpath(args.ckpt, _ROOT), "sha256": ck_sha},
        "warm_start_m7dj_sha256": m7["sha256"],
        "c3b_split_hashes": man["split"]["hashes"],
        "asserts": {"pool2full_index_map_max_abs_err": map_err,
                    "bank_mean_vs_manifest_max_abs_err": d_mu,
                    "adapter_pairs_space_identical_to_c3b": space_same,
                    "blend_target_rebuild_max_abs_diff": d_tgt,
                    "concept_dirs_reproduced_worst_1_minus_abscos": br.dc_reproduction_worst},
        "what_may_not_be_claimed": (
            "NO VALUE CLAIM: every NDCG row runs under a declared-circular answer model and carries "
            "at most a RELATIVE reading (AUTHOR RULING 2026-08-25) - nothing at states outside "
            "C1-final's 57 - nothing about the item channel - not that the questions are good "
            "because they read well - not that a human would answer them"),
        "git": GB,
        "generation_diagnostics": {f"{sp}/{a}": gdiag[(sp, a)] for sp in SPACES for a in DEC_ARMS
                                   if (sp, a) in gdiag},
        "turns": []}

    def flush():
        """HARD RULE 11a: every turn is on disk the moment it completes."""
        out["minutes"] = round((time.time() - t00) / 60, 2)
        out["turns"] = rowsheet
        with open(OUT_JSON + ".tmp", "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=1, default=float)
        os.replace(OUT_JSON + ".tmp", OUT_JSON)
        savez_atomic(OUT_NPZ, **npz_out)

    flush()
    for k, t in enumerate(turns):
        rws = [int(r) for r in t["rows"]]
        si = t["state_idx"]
        row = {"turn": t["turn"], "sid": t["sid"], "depth": t["depth"], "n_users": len(rws),
               "emits": bool(t["emits"]),
               "catalogue": {"question": t["catalogue_question"], "phrase": t["catalogue_phrase"],
                             "channel": t["catalogue_channel"], "splitter": list(t["splitter"]),
                             "splitter_gain": t["splitter_gain"],
                             "behavioural_answer_rate": t["behavioural_answer_rate"],
                             "branch_counts": t["branch_counts"],
                             "branch_taken": t["branch_taken"]},
               "synthesised": None,
               "no_emission_marker": None if t["emits"] else
                                    "[no emission -- outside the fitted set]"}
        # ---- the synthesised question at this turn ----------------------------------------------
        ridx = np.asarray([rowpos[r] for r in rws], np.int64)
        if t["emits"]:
            j = emit_turn_ids.index(t["turn"])
            sel = sel_all[(args.space, DEC_PRIMARY)][j]
            cids = sorted(int(c) for c, _v in sel)
            L = [int(c) for c in np.argsort(-cos_tower[si], kind="stable")[:TOPK_ASK]]
            cu = float(C3B.rowcos(U_prim[j:j + 1], R[si:si + 1])[0])
            cu = None if not np.isfinite(cu) else cu
            anch = POC.top_anchors(Zemit[si, :-1] - m7["muK"], K7u, br.names, ni, k=5)
            ansfrac = {}
            for c in cids:
                ansfrac[br.tag_names[c]] = float((sgn[ridx, c] > 0).mean())
            row["synthesised"] = {
                "space": args.space, "space_renders": sp_diag[args.space]["renders"],
                "text": texts[(args.space, DEC_PRIMARY)][j],
                "text_greedy": texts[(args.space, "greedy")][j],
                "text_all_spaces": {sp: {a: texts[(sp, a)][j] for a in DEC_ARMS} for sp in SPACES},
                "read_back_concepts_all_spaces": {
                    sp: [br.tag_names[int(c)] for c, _v in sel_all[(sp, DEC_PRIMARY)][j]]
                    for sp in SPACES},
                "decoding": DEC_PRIMARY,
                "emitted_direction_nearest_bank_anchors": anch,
                "emitted_cos_to_own_blend_target": float(cos_own[si]),
                "read_back_concepts": [br.tag_names[c] for c in cids],
                "read_back_cos_to_R_s": cu,
                "L_arm_top4_by_cos_to_R_s": [br.tag_names[c] for c in L],
                "overlap_with_L_arm": int(len(set(cids) & set(L))),
                "cohort_answer_" + CIRC.replace(" ", "_"): ansfrac,
                "n_concepts_folded": len(cids)}
            for c in cids:
                if (1, c) not in syn_seq:
                    syn_seq.append((1, c))
        tree_seq.append((int(t["splitter"][0]), int(t["splitter"][1])))

        # ---- the fold, the NDCG and the recommendations, both columns on THIS cohort -------------
        bf, bt_ = genv_eval(E, [], rws)
        cells = {"base_m0": {"full": float(np.nanmean(bf)), "tail": float(np.nanmean(bt_))}}
        recs = {}
        m_base = mean_scores(E, [], rws)
        recs["base_m0"] = top_items(m_base, br.names, TOPK_REC)
        for nm, seq in (("synthesised", syn_seq), ("tree", tree_seq)):
            if not seq:
                cells[nm] = None
                recs[nm] = None
                continue
            f, tl = genv_eval(E, list(seq), rws)
            cells[nm] = {"full": float(np.nanmean(f)), "tail": float(np.nanmean(tl)),
                         "minus_base_full": float(np.nanmean(f) - np.nanmean(bf)),
                         "minus_base_tail": float(np.nanmean(tl) - np.nanmean(bt_)),
                         "n_questions_folded": len(seq)}
            # the risers are measured against THIS cohort's own m=0 base, so the column shows what
            # the interview so far bought at this state -- not a drift between two cohorts
            recs[nm] = top_items(mean_scores(E, list(seq), rws), br.names, TOPK_REC, prev=m_base)
        row["levels_" + CIRC.replace(" ", "_")] = cells
        row["top5_recommendations_display_only"] = recs
        row["cumulative_folded"] = {
            "synthesised_concepts": [br.tag_names[c] for _ch, c in syn_seq],
            "tree_questions": [POC.phrase(br.names, ni, int(pool2full[pool_index[q]]))
                               for q in tree_seq]}
        rowsheet.append(row)
        npz_out[f"t{k}_rows"] = np.asarray(rws, np.int64)
        npz_out[f"t{k}_base_full"] = bf.astype(np.float32)
        npz_out[f"t{k}_base_tail"] = bt_.astype(np.float32)
        flush()
        syn_f = "--" if cells["synthesised"] is None else f"{cells['synthesised']['full']:.5f}"
        log(f"  [turn {t['turn']}] {t['sid']:<34} n={len(rws):>5} "
            f"{'EMIT' if t['emits'] else 'no-emission':<11} syn {syn_f} "
            f"tree {cells['tree']['full']:.5f} base {cells['base_m0']['full']:.5f}")

    # ============================================================ output
    flush()
    write_atomic(OUT_MD, markdown(out))
    write_atomic(OUT_TEX, latex(out))
    log("=" * 110)
    for r in rowsheet:
        s = r["synthesised"]
        log(f"  turn {r['turn']} {r['sid']:<34} n={r['n_users']:>5}")
        log(f"    tree : {r['catalogue']['question']}")
        log(f"    synth: {(s['text'] if s else r['no_emission_marker'])!r}")
    log("=" * 110)
    log(f"[done] {OUT_JSON}\n       {OUT_MD}\n       {OUT_TEX}\n       {OUT_NPZ}  "
        f"({out['minutes']:.1f}m)")
    return out


# ============================================================================ the two tables
def _f(x, n=5):
    return "--" if x is None else f"{x:.{n}f}"


def _md(s):
    """A cell of a markdown table: a pipe or a newline in a generated question would silently break
    the table, so both are neutralised rather than trusted."""
    return str(s).replace("|", r"\|").replace("\r", " ").replace("\n", " ")


def markdown(out):
    L = ["# C4D -- the synthesised-interview transcript",
         "",
         f"*{out['what_this_is']}.*", "",
         f"- adapter: `{out['renderer']['adapter']}` ({out['renderer']['targets_mode']} targets), "
         f"decoding **{out['renderer']['decoding_primary']}** (greedy shown beside it)",
         f"- emission space: **{out['space']['chosen']}** -- "
         f"{out['space']['constructions'].get(out['space']['chosen'], {}).get('renders', '?')}",
         f"- cohort: {out['walk']['cohort_n_users']:,} validation users, path mode "
         f"`{out['walk']['path_mode']}`; stop: {out['walk']['stop_reason']}",
         f"- answers, folds and every NDCG below are **{out['answer_model']['label']}** and carry no "
         f"value claim",
         ""]
    sp = out["space"]["probe"].get(out["space"]["chosen"])
    if sp:
        pr = out["space"]["probe"]
        L += [f"> ⚠ SPACE PROBE (why this row is rendered in the `{out['space']['chosen']}` "
              f"construction). EXACT top-1 agreement between the fed point's nearest bank entry and "
              f"the emission's own nearest entry in m7dj's space is "
              f"**{sp['snap_agreement_with_the_emission_in_its_own_space']:.3f}** — and it is "
              f"0.000 for ALL THREE constructions, including `target`, which is the right direction "
              f"in the adapter's own bank by construction. Exact-index equality over a "
              f"{N_BANK:,}-entry bank is therefore too strict to separate them; the arm that does "
              f"is the NEIGHBOURHOOD overlap (how many of the emission's five nearest bank entries "
              f"the fed point also names among its five, chance "
              f"{sp['top5_neighbourhood_overlap_chance']:.5f}): "
              + ", ".join(f"**{k} {pr[k]['top5_neighbourhood_overlap_mean']:.2f}/5**"
                          for k in out["space"]["all_built"] if k in pr)
              + ". The two policies' k(phi) read-outs do not share a basis, so a `direct` render "
                "would be a render of a DIFFERENT direction; the `target` construction renders the "
                "state's measured-gain blend rebuilt in the renderer's own bank, and C1-final "
                "reports cos(emission, target) = 0.970 at exactly these trained states.", ""]
    L += ["| turn | state | n | synthesised question (policy's own direction, in English) | "
          "the catalogue question the tree would ask | folded | full@10 | tail@10 | top-5 after "
          "this turn |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in out["turns"]:
        s = r["synthesised"]
        cell = r["levels_" + out["answer_model"]["label"].replace(" ", "_")]
        syn = (f"**{_md(s['text'])}**<br/>*greedy:* {_md(s['text_greedy'])}<br/>read back: "
               f"{_md(', '.join(s['read_back_concepts']))}" if s else
               f"*{r['no_emission_marker']}*")
        fold = _md(", ".join(r["cumulative_folded"]["synthesised_concepts"])) or "--"
        c = cell["synthesised"]
        recs = r["top5_recommendations_display_only"]["synthesised"]
        top = _md("; ".join(x["title"] for x in (recs or []) if "rank" in x)) or "--"
        L.append(f"| {r['turn']} | `{r['sid']}` | {r['n_users']:,} | {syn} | "
                 f"{_md(r['catalogue']['question'])} | {fold} | "
                 f"{_f(c['full'] if c else None)} | {_f(c['tail'] if c else None)} | {top} |")
    L += ["", "### the same turns, the TREE's own column (the honest comparator)", "",
          "| turn | state | tree question | full@10 | tail@10 | base (m=0) full |",
          "|---|---|---|---|---|---|"]
    for r in out["turns"]:
        cell = r["levels_" + out["answer_model"]["label"].replace(" ", "_")]
        L.append(f"| {r['turn']} | `{r['sid']}` | {_md(r['catalogue']['question'])} | "
                 f"{_f(cell['tree']['full'])} | {_f(cell['tree']['tail'])} | "
                 f"{_f(cell['base_m0']['full'])} |")
    L += ["", "---", "", f"**What this may not claim.** {out['what_may_not_be_claimed']}.", ""]
    return "\n".join(L)


def _tex(s):
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"),
                 ("#", r"\#"), ("_", r"\_"), ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"),
                 ("^", r"\textasciicircum{}")):
        s = str(s).replace(a, b)
    return s


def latex(out):
    L = [r"% C4D -- the synthesised-interview transcript. Self-contained fragment: needs only",
         r"% booktabs. Produced by src/instrument/c4d_transcript.py -- DO NOT EDIT BY HAND.",
         r"\begin{table*}[t]", r"\centering", r"\small",
         r"\begin{tabular}{@{}rlp{0.30\linewidth}p{0.26\linewidth}rr@{}}", r"\toprule",
         r"\# & state & synthesised question (the policy's own direction, rendered) & "
         r"the catalogue question the tree would ask & full@10 & tail@10 \\", r"\midrule"]
    lab = out["answer_model"]["label"].replace(" ", "_")
    for r in out["turns"]:
        s = r["synthesised"]
        c = r["levels_" + lab]["synthesised"]
        syn = (r"\emph{" + _tex(s["text"]) + "}" if s else
               r"\textsc{" + _tex(r["no_emission_marker"].strip("[]")) + "}")
        L.append(f"{r['turn']} & \\texttt{{{_tex(r['sid'])}}} & {syn} & "
                 f"{_tex(r['catalogue']['question'])} & "
                 f"{_f(c['full'] if c else None, 4)} & {_f(c['tail'] if c else None, 4)} \\\\")
    L += [r"\bottomrule", r"\end{tabular}",
          r"\caption{" + _tex(
              f"One cold-start interview. At every state where the C1-final policy was trained to "
              f"emit ({out['n_turns_emitting']} of {out['n_turns']} turns) the question is the "
              f"policy's own off-catalogue direction, rendered by the adapter under "
              f"{out['renderer']['decoding_primary']}; where it was not, no question is invented and "
              f"the row says so. Answers, folds and both NDCG columns are "
              f"{out['answer_model']['label']} and carry no value claim.") + r"}",
          r"\label{tab:c4d-transcript}", r"\end{table*}"]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description="Paper C / C4D -- the synthesised-interview transcript "
                                             "(docs/results/C4D_TRANSCRIPT_SPEC.md)")
    ap.add_argument("--ckpt", default=E1_CKPT)
    ap.add_argument("--snapshot", default=None)
    ap.add_argument("--concept_ckpt", default=None)
    ap.add_argument("--adapter", default="v2", choices=["phrases", "phrase", "v2", "questions"],
                    help="which C3B adapter renders: `phrases` = the bare-phrase C3B adapter, "
                         "`v2`/`questions` = the c3b_v2 question adapter")
    ap.add_argument("--space", default="direct", choices=list(SPACES),
                    help="how C1-final's emission enters the adapter's input space (all three are "
                         "built and rendered; this picks the one the tables show)")
    ap.add_argument("--channel", default="concept", choices=["concept", "item"])
    ap.add_argument("--path", default="majority",
                    help="'majority' (largest fitted child), 'emitting' (largest child that is a "
                         "C1-final trained state, else majority), or an explicit 'root.dislike.like'")
    ap.add_argument("--n_users", type=int, default=0,
                    help="0 = the whole validation cohort (default, no truncation); a positive N "
                         "takes the leading N rows and DECLARES it in the output")
    ap.add_argument("--max_turns", type=int, default=6)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--gen_batch", type=int, default=64)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--allow_dirty", action="store_true",
                    help="EMERGENCY ONLY: run with uncommitted IMPORTED src/instrument code")
    args = ap.parse_args()
    args.threads = min(int(args.threads), 4)
    from strategy_channel_suite import CERT_TOWER, CERT_CONC
    args.snapshot = args.snapshot or CERT_TOWER
    args.concept_ckpt = args.concept_ckpt or CERT_CONC
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    torch.set_num_threads(int(args.threads))
    log("#" * 110)
    log("# PAPER C / C4D -- THE SYNTHESISED-INTERVIEW TRANSCRIPT. Emission only at C1-final's 57 "
        "TRAINED states; a state outside that set gets the tree's own question and a "
        "'[no emission]' marker. Answers are simulated and DECLARED CIRCULAR; no value claim. "
        "VALIDATION ONLY: the 10,000 TEST users are not built, not loaded, not touched. No LLM API.")
    log("#" * 110)
    return run(args)


if __name__ == "__main__":
    main()
