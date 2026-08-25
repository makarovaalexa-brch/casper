r"""PAPER C / STAGE C4B -- TRANSLATION NOISE: how much of the DIRECTION's contribution survives
being said in English and read back?

Design sheet: `docs/design/DS_C4B_TRANSLATION_NOISE.md` (PRE-REGISTERED 2026-08-25, BINDING) under
the AUTHOR RULING 2026-08-25 recorded at the top of `docs/design/PAPER_C_DESIGN.md`: a simulated
geometric answer model is ADMITTED, **declared circular in the paper's own voice**, and may carry a
**RELATIVE claim only** -- the translation cost. No absolute "asking it helps" claim routes through
this file. Every absolute row printed here is labelled *"under a declared-circular answer model"*.

THE FOUR ARMS (per state; the C1 pilot's registered 42; users = the state's OWN cohort rows)
--------------------------------------------------------------------------------------------
  L        the DIRECTION itself: the top-4 concepts by cos(d_c, R_s) in tower space, ties by
           concept id. Pure geometry, no text anywhere in the arm.
  T        the TRANSLATION: the SBERT top-4 (mincos 0.30) of adapter A's banked concept-hot render
           of that state's m=5 BLEND TARGET, read from `c4_pilot_per_state.npz` -- never regenerated.
  T'       the same, from the scrambled twin A' (the correspondence-destroying control).
  P        the same, from the channel-matched PRIOR-ONLY text (zeroed latent + concept one-hot).

  Cost = L - T, both arms under the SAME simulator, so the circular component is common-mode and the
  difference is the verbalisation noise -- the quantity the author asked for.

THE ANSWER MODEL (declared circular; the June cans convention, cited beside Info-Synth's geometric
response model and PEBOL's simulated users in the paper)
------------------------------------------------------------------------------------------------
  answer(user r, concept c) = sign( u*_r . d_c ),  u*_r = the tower fold of the user's FULL FOLD-IN
  PROFILE (`fold_items_frozen(ctx.enc, ctx.allb[r])`, exactly `answer_contrast`'s `u_star`) and d_c
  the env's own whitened concept direction. The held-out targets are not in `ctx.allb` and that is
  ASSERTED (`leak_assert`), so the simulator is circular but never clairvoyant. Refusal is NEVER
  simulated: every asked concept is answerable, at +-1.

  The answers enter the CERTIFIED path unchanged: `E["V"]` is replaced by the +-1 sign matrix and
  `E["foldable"]` by all-True, and the fold is then the env's own -- `genv_eval` -> `genv_latent` ->
  the certified `ConceptFoldNet(cv3)`, `cap_negatives` over the REVEALED set, policy-B pool, learned
  decoder bias, `ndcg10_from_scores` with the mask contract live. NO SECOND SCORER EXISTS HERE.

TWO DECLARED CHOICES (neither is fixed by the sheet; both are arm-symmetric, so neither can move a
difference; both are recorded in the output JSON)
  (i)  the arms are scored AGAINST THE m=0 (NO-FOLD) BASE, as the sheet's base row says -- the
       state's own interview PREFIX is not folded in any arm. The state supplies the DIRECTION and
       the COHORT; the four arms differ in exactly the four concepts asked.
  (ii) within an arm the concepts are asked in ASCENDING CONCEPT ID, a fixed arm-symmetric ask order,
       so that two arms selecting the SAME set produce the identical fold (which is what makes
       reading 4 -- the overlap accounting -- readable at all).

WHAT MAY NOT BE CLAIMED. Absolute NDCG value of asking anything (the rows are printed and labelled,
never headlined) - anything about the item channel - anything from emission renders (retracted;
C4-final re-renders after the declared rescale) - generalisation beyond the 42 nested states.
VALIDATION ONLY: the 10,000 canonical TEST users are not built, not loaded, not touched. No LLM API.
"""
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

from policy_net import Policy, build_phi, load_traj, log, SEED
from c0_geometry import (PAPERC, E1_CKPT, N_BANK, VAL_M0_FULL, VAL_M0_TAIL, sha256, git, git_block,
                         savez_atomic, index_map_assert, bank_vectors, dist)
import c1_pilot as C1
import c3_poc_extract as POC
import c3b_adapter as C3B
import c4_pilot as C4

OUT_JSON = os.path.join(PAPERC, "c4b_noise.json")
OUT_NPZ = os.path.join(PAPERC, "c4b_noise_per_state.npz")
IN_NPZ = os.path.join(PAPERC, "c4_pilot_per_state.npz")
IN_JSON = os.path.join(PAPERC, "c4_pilot.json")

M = C4.M                                  # 5, the pilot's primary blend width
N_STATES = C4.N_STATES                    # 42, the registered population
TOPK = 4                                  # arm L takes FOUR concepts, matching the SBERT top-4
BOOT_SEED = SEED                          # 98765, the registered bootstrap seed
N_BOOT = C3B.N_BOOT                       # 10,000 resamples
BOOT_CHUNK = 200                          # replicate-block size; see boot_mean_chunked
ARMS = ("L", "T", "Tprime", "P")
CELL_A = ("target", "concept_hot", "A")
CELL_AP = ("target", "concept_hot", "Aprime")
PRIOR_KEY = "text_prior_concept_hot_A"
CIRC = "under a declared-circular answer model"


# ============================================================================ statistics
def boot_mean_chunked(d, n=N_BOOT, seed=BOOT_SEED, chunk=BOOT_CHUNK):
    """`c3b_adapter.boot_mean`, drawn in replicate BLOCKS instead of one array.

    Identical numbers, not an approximation: `Generator.integers` fills an array from the stream in
    C order, so 50 draws of (200, n) consume exactly the stream that one draw of (10000, n) does.
    `boot_mean_selftest` asserts that at runtime rather than leaving it as this comment. The reason
    it has to exist: one of the 42 cohorts is the whole 10,000-user validation set, and boot_mean's
    single (10000, 10000) float64 array is 800 MB -- an OOM in the middle of the run (HARD RULE 11d).
    """
    d = np.asarray(d, np.float64)
    d = d[np.isfinite(d)]
    if not len(d):
        return {"mean": None, "ci_lo": None, "ci_hi": None, "n": 0, "n_boot": int(n),
                "excludes_zero": False}
    rng = np.random.default_rng(seed)
    m = np.empty(n, np.float64)
    for st in range(0, n, chunk):
        k = min(chunk, n - st)
        m[st:st + k] = d[rng.integers(0, len(d), size=(k, len(d)))].mean(1)
    lo, hi = np.percentile(m, [2.5, 97.5])
    return {"mean": float(d.mean()), "ci_lo": float(lo), "ci_hi": float(hi),
            "n": int(len(d)), "n_boot": int(n), "excludes_zero": bool(lo > 0 or hi < 0)}


def boot_mean_selftest(n=400):
    """The chunked bootstrap must equal the banked `c3b_adapter.boot_mean` EXACTLY."""
    v = np.random.default_rng(7).standard_normal(n)
    a = C3B.boot_mean(v, n=N_BOOT, seed=BOOT_SEED)
    b = boot_mean_chunked(v, n=N_BOOT, seed=BOOT_SEED)
    for k in ("mean", "ci_lo", "ci_hi", "n"):
        assert a[k] == b[k], f"chunked bootstrap != c3b_adapter.boot_mean at {k}: {a[k]} vs {b[k]}"
    return {"n_selftest": int(n), "mean": a["mean"], "ci_lo": a["ci_lo"], "ci_hi": a["ci_hi"]}


def directional(b):
    """A directional reading passes only when the WHOLE interval sits above zero (the C2 lesson)."""
    return bool(b["mean"] is not None and b["ci_lo"] is not None and b["ci_lo"] > 0)


class StratBoot:
    """The cohort-weighted summary: users resampled WITHIN state, states held fixed.

    DECLARED AGGREGATION UNIT: user-rows within state, states nested. The weight is the state's own
    cohort size, so the summary is the pooled per-user-row mean; resampling inside the state and
    never across states is what makes it a summary OF the 42 states rather than of a pretend
    population of states. Every registered reading and every absolute level is carried through ONE
    pass of resample indices per state, so all of them share the same 10,000 replicates -- otherwise
    the 36,197 user-rows would be redrawn sixteen times for nothing.

    THE CAVEAT THAT TRAVELS: the 42 states are NESTED, so a user can appear in several of them. This
    bootstrap treats the states as independent strata and therefore UNDERSTATES the summary interval.
    The per-state rows, which are clean, are the primary object; the summary is the weighted read.
    """

    def __init__(self, keys, seed=BOOT_SEED, n=N_BOOT, chunk=BOOT_CHUNK):
        self.keys = list(keys)
        self.acc = {k: np.zeros(n, np.float64) for k in self.keys}
        self.num = {k: 0.0 for k in self.keys}
        self.den = {k: 0.0 for k in self.keys}
        self.rng = np.random.default_rng(seed)
        self.n = n
        self.chunk = chunk

    def add_state(self, vals):
        """`vals`: {key -> 1-D array}. Arrays sharing a length share their resample indices; a key
        whose array is empty contributes nothing to that key (and to no other)."""
        groups = {}
        for k, v in vals.items():
            v = np.asarray(v, np.float64)
            assert np.isfinite(v).all(), "StratBoot takes finite arrays only (mask before adding)"
            if not len(v):
                continue
            groups.setdefault(len(v), []).append((k, v))
        for ln, items in sorted(groups.items()):
            for k, v in items:
                self.num[k] += float(v.sum())
                self.den[k] += float(ln)
            for st in range(0, self.n, self.chunk):
                kk = min(self.chunk, self.n - st)
                idx = self.rng.integers(0, ln, size=(kk, ln))
                for k, v in items:
                    self.acc[k][st:st + kk] += float(ln) * v[idx].mean(1)

    def read(self, key):
        if self.den[key] <= 0:
            return {"mean": None, "ci_lo": None, "ci_hi": None, "n_user_rows": 0,
                    "n_boot": int(self.n)}
        m = self.acc[key] / self.den[key]
        lo, hi = np.percentile(m, [2.5, 97.5])
        return {"mean": float(self.num[key] / self.den[key]), "ci_lo": float(lo),
                "ci_hi": float(hi), "n_user_rows": int(round(self.den[key])),
                "n_boot": int(self.n), "excludes_zero": bool(lo > 0 or hi < 0),
                "unit": "user-rows within state, states nested (weight = cohort size)"}


# ============================================================================ the answer model
def geometric_answers(br, rows):
    """The DECLARED-CIRCULAR answer model: answer(r, c) = sign(u*_r . d_c), at +-1.

    `u*_r` is the tower fold of the user's whole FOLD-IN profile -- `answer_contrast.py`'s `u_star`,
    built from `ctx.allb` (held-out targets removed, asserted by `leak_assert`). Refusal is never
    simulated. An exact zero projection cannot be signed; it is broken to +1 and COUNTED, never
    silently absorbed.
    """
    from concept_fold import fold_items_frozen
    ctx = br.E["ctx"]
    seqs = [ctx.allb[int(r)] for r in rows]
    U = fold_items_frozen(ctx.enc, seqs).numpy().astype(np.float64)
    assert U.shape == (len(rows), ctx.d) and np.isfinite(U).all()
    proj = U @ br.d_c.T
    assert proj.shape == (len(rows), br.C)
    n_zero = int((proj == 0.0).sum())
    sgn = np.where(proj >= 0.0, 1.0, -1.0).astype(np.float32)
    diag = {"rule": "answer(r, c) = sign(u*_r . d_c), +-1, refusal never simulated",
            "u_star": ("fold_items_frozen(ctx.enc, ctx.allb[r]) -- the env's own full fold-in "
                       "profile latent, identical in construction to answer_contrast.u_star"),
            "n_user_concept_cells": int(proj.size),
            "n_exact_zero_projections_broken_to_plus1": n_zero,
            "frac_positive": float((sgn > 0).mean()),
            "abs_projection": dist(np.abs(proj).ravel()),          # every cell, not a sample
            "declared_circular": True}
    return sgn, U, diag


def install_answer_model(E, rows, sgn):
    """Replace the env's behavioural answer table with the declared-circular one, IN PLACE.

    Nothing else about the environment moves: the pool, the decoder bias, the head mask, the concept
    net and the negative cap are the certified ones, and `genv_eval` is still the only funnel. Rows
    outside the 42 cohorts are left un-answerable so that a stray row could never be scored silently.
    """
    ctx = E["ctx"]
    V = np.zeros((ctx.n, E["C"]), np.float32)
    F = np.zeros((ctx.n, E["C"]), bool)
    a = np.asarray(rows, np.int64)
    V[a] = sgn
    F[a] = True
    old_V, old_F = E["V"], E["foldable"]
    assert old_V.shape == V.shape and old_F.shape == F.shape, "the answer table changed shape"
    E["V"], E["foldable"] = V, F
    return {"n_rows_answerable": int(len(a)),
            "behavioural_answer_rate_replaced": float(old_F[a].mean()),
            "geometric_answer_rate": float(F[a].mean()),
            "negative_cap_c_neg": float(E["cneg"]),
            "note": ("the certified fold's cap_negatives over the REVEALED set still applies, so a "
                     "user answering k of the 4 concepts negatively folds them at -min(1, c_neg/k) "
                     "rather than -1. It is the env's own certified operator and it is applied "
                     "identically in all four arms.")}


# ============================================================================ the run
def run(args):
    from ndcg_tree import build_pool, leak_assert
    from strategy_channel_suite import genv_mean, genv_eval, contract_selftest
    t00 = time.time()
    os.makedirs(PAPERC, exist_ok=True)
    GB = git_block(allow_dirty=args.allow_dirty, threads=args.threads,
                   extra={"status_short": git("status", "--short")})     # HARD RULE 10, first
    log(f"[hardrule10] src/instrument clean={GB['src_instrument_clean']} rev={GB['rev'][:8]} "
        f"threads={GB['threads']} torch_threads={GB['torch_num_threads']}")
    bt = boot_mean_selftest()
    log(f"[selftest] chunked bootstrap == c3b_adapter.boot_mean exactly (mean {bt['mean']:+.6f})")

    # ---- the bridge (env + tower targets + SBERT), exactly as the pilot builds it ----------------
    br = C3B.TowerBridge(args)
    E, ni, C = br.E, br.ni, br.C
    blob = torch.load(args.ckpt, map_location="cpu")
    ck_sha = sha256(args.ckpt)
    assert ck_sha == C1.C0_CKPT_SHA == br.ckpt_sha

    # HARD RULE 5, BEFORE ANYTHING: this is the certified ruler, not a lookalike env
    f0, t0_ = genv_mean(E, [], E["rows"])
    assert round(f0, 8) == round(VAL_M0_FULL, 8), f"canonical snap FAILED (full): {f0!r}"
    assert round(t0_, 8) == round(VAL_M0_TAIL, 8), f"canonical snap FAILED (tail): {t0_!r}"
    log(f"[gate] validation m=0 intercept full={f0!r} tail={t0_!r} -- both canonical")
    contract_selftest(E, E["rows"])
    leak_assert(E, E["rows"], "paperC/c4b")

    pool, _it, _cc = build_pool(E, 500, 500)
    pool = [(int(a), int(b)) for a, b in pool]
    assert pool == [(int(a), int(b)) for a, b in blob["pool"]], "the env's pool is not the ckpt's"
    pool_index = {q: i for i, q in enumerate(pool)}
    assert ni + C == N_BANK, f"bank composition {ni}+{C} != {N_BANK}"
    pool2full = np.asarray([(i if ch == 0 else ni + i) for ch, i in pool], np.int64)

    phi_t, _ = build_phi(E, pool, norm=(blob["norm_mu"], blob["norm_sd"]))
    pol = Policy(E["ctx"].d, phi_t.shape[1], N_BANK, arch=blob["arch"], state=blob["state"],
                 use_pos=blob["use_pos"])
    pol.load_state_dict(blob["model"]); pol.eval()
    Epool, Kpool = bank_vectors(pol, phi_t)
    zp = np.load(C3B.PAIRS_NPZ)
    Kbank = np.asarray(zp["K"], np.float64)
    map_err = index_map_assert(Kbank, pool2full, Kpool.astype(np.float64))

    # ---- the 42 states and their m=5 blends -------------------------------------------------------
    Dva = load_traj(train=False)
    reg, extra = C1.load_the_42(pool_index, Dva)
    states = reg + extra
    assert len(reg) == N_STATES
    for s in states:
        s["pool_prefix_idx"] = [pool_index[q] for q in s["prefix"]]
    Tt, TKt, meta = C1.build_targets(states, Epool, Kpool, pool_index, M)
    d_tgt = float(np.abs(Tt - np.load(os.path.join(PAPERC, "c1_pilot_per_state.npz"))
                         [f"target_m{M}"]).max())
    assert d_tgt == 0.0, f"the rebuilt m={M} targets are not the pilot's (max|diff| {d_tgt:g})"

    # ---- PROVENANCE ON THE RENDERS: the npz and the json must be ONE object ------------------------
    npz = np.load(IN_NPZ, allow_pickle=True)
    with open(IN_JSON, encoding="utf-8") as fh:
        c4j = json.load(fh)
    assert c4j["analysis"] == "paperC_C4_pilot"
    assert c4j["checkpoint"]["sha256"] == ck_sha, "c4_pilot.json is on another checkpoint"
    assert c4j["c3b_split_hashes"] == C3B.load_manifest()["split"]["hashes"], \
        "the C4 renders were produced under a DIFFERENT C3B split than this manifest's"
    d_pt = float(np.abs(TKt[:N_STATES].astype(np.float32)
                        - np.asarray(npz["point_target"], np.float32)).max())
    assert d_pt == 0.0, f"the rebuilt target points are not the C4 pilot's (max|diff| {d_pt:g})"
    txt_A = [str(x) for x in npz["text_" + "_".join(CELL_A)]]
    txt_AP = [str(x) for x in npz["text_" + "_".join(CELL_AP)]]
    prior_txt = str(np.asarray(npz[PRIOR_KEY]).ravel()[0])
    assert len(txt_A) == len(txt_AP) == N_STATES
    # the npz's texts and cosines must BE the ones the json's demo table reports, row by row
    dem = c4j["demo_table"]
    assert len(dem) == N_STATES
    n_txt_bad = 0
    for i, row in enumerate(dem):
        assert row["sid"] == reg[i]["sid"], f"demo table row {i} is not state {reg[i]['sid']}"
        for cell, tx in (("/".join(CELL_A), txt_A[i]), ("/".join(CELL_AP), txt_AP[i])):
            if row["renders"][cell]["text"] != tx:
                n_txt_bad += 1
    assert n_txt_bad == 0, f"{n_txt_bad} banked render(s) disagree between c4_pilot.json and its npz"
    assert c4j["prior_only_texts"]["A"]["concept_hot"] == prior_txt, \
        "the prior-only concept-hot text disagrees between c4_pilot.json and its npz"
    for cell in ("/".join(CELL_A), "/".join(CELL_AP)):
        j = np.asarray([np.nan if r["renders"][cell]["cos_to_R"] is None
                        else r["renders"][cell]["cos_to_R"] for r in dem], np.float64)
        n = np.asarray(npz["cos_" + cell.replace("/", "_")], np.float64)
        m = np.isfinite(j) & np.isfinite(n)
        assert (np.isfinite(j) == np.isfinite(n)).all() and float(np.abs(j[m] - n[m]).max()) < 1e-12,\
            f"the banked cos_to_R for {cell} disagrees between c4_pilot.json and its npz"
    render_prov = {"npz": os.path.relpath(IN_NPZ, _ROOT), "npz_sha256": sha256(IN_NPZ),
                   "json": os.path.relpath(IN_JSON, _ROOT), "json_sha256": sha256(IN_JSON),
                   "c4_git_rev": c4j["git"]["rev"], "c4_git_branch": c4j["git"].get("branch"),
                   "c4_src_instrument_clean": c4j["git"].get("src_instrument_clean"),
                   "c4_checkpoint_sha256": c4j["checkpoint"]["sha256"],
                   "c1_emission_provenance": c4j["asserts"]["c1_emission_provenance"],
                   "cells_read": ["/".join(CELL_A), "/".join(CELL_AP), "prior/concept_hot/A"],
                   "renders_regenerated": False}
    log(f"[gate] renders READ from {os.path.basename(IN_NPZ)} (C4 rev {c4j['git']['rev'][:8]}, "
        f"src clean={c4j['git'].get('src_instrument_clean')}); texts and cosines agree with "
        f"c4_pilot.json row by row; prior text {prior_txt!r}")

    # ---- R_s: IMPORTED from the pilot, then asserted bit-identical to the banked R ------------------
    R, r_disc = C4.reference_directions(br, meta, pool, pool2full)
    d_R = float(np.abs(R - np.asarray(npz["R"], np.float64)).max())
    assert d_R == 0.0, f"the rebuilt R_s is not the pilot's banked R (max|diff| {d_R:g})"
    log(f"[reference] R_s {R.shape} rebuilt bit-identically to the pilot's banked R")

    # ---- the four arms' concept sets ----------------------------------------------------------------
    D_unit = br.d_c / np.linalg.norm(br.d_c, axis=1, keepdims=True)
    R_unit = R / np.linalg.norm(R, axis=1, keepdims=True)
    cos_tower = R_unit @ D_unit.T                                   # (42, C)
    # ties by concept id: lexsort on (-cos, id) via a stable sort of the descending cosine
    arm_c = {a: [] for a in ARMS}
    arm_meta = {a: [] for a in ARMS}
    for i in range(N_STATES):
        o = np.argsort(-cos_tower[i], kind="stable")[:TOPK]         # stable => ties by concept id
        arm_c["L"].append(sorted(int(c) for c in o))
        arm_meta["L"].append([{"concept": br.tag_names[int(c)], "cos_to_R": float(cos_tower[i, c])}
                              for c in o])
    sel_A = br.select(txt_A)
    sel_AP = br.select(txt_AP)
    sel_P = br.select([prior_txt] * N_STATES)
    for name, sel in (("T", sel_A), ("Tprime", sel_AP), ("P", sel_P)):
        for i in range(N_STATES):
            arm_c[name].append(sorted(int(c) for c, _v in sel[i]))
            arm_meta[name].append([{"concept": br.tag_names[int(c)], "sbert_cos": float(v)}
                                   for c, v in sel[i]])
    for a in ARMS:
        assert len(arm_c[a]) == N_STATES
        assert all(len(x) == len(set(x)) <= TOPK for x in arm_c[a])
    overlap_LT = np.asarray([len(set(arm_c["L"][i]) & set(arm_c["T"][i])) for i in range(N_STATES)],
                            np.int64)
    log(f"[arms] |concepts| per state: " + ", ".join(
        f"{a} {np.mean([len(x) for x in arm_c[a]]):.2f}" for a in ARMS)
        + f" | |L n T| mean {overlap_LT.mean():.3f} "
          f"(hist {dict(zip(*[x.tolist() for x in np.unique(overlap_LT, return_counts=True)]))})")

    # ---- the declared-circular answer model, installed into the certified env ---------------------
    rows_all = sorted({int(r) for s in reg for r in s["rows"]})
    act = set(int(r) for r in E["rows"])
    miss = [r for r in rows_all if r not in act]
    assert not miss, f"{len(miss)} state cohort rows are outside the env's active cohort"
    sgn, U_star, ans_diag = geometric_answers(br, rows_all)
    ins = install_answer_model(E, rows_all, sgn)
    row_pos = {r: j for j, r in enumerate(rows_all)}
    log(f"[answers] declared-circular sign rule installed over {len(rows_all):,} distinct user-rows "
        f"({ans_diag['frac_positive'] * 100:.2f}% positive, "
        f"{ans_diag['n_exact_zero_projections_broken_to_plus1']} exact zeros); the behavioural "
        f"answer table it replaces had answer rate {ins['behavioural_answer_rate_replaced']:.4f}")

    # ---- per-state scoring -------------------------------------------------------------------------
    PAIRS = (("L", "T"), ("T", "Tprime"), ("T", "P"))
    LEVELS = ("base",) + ARMS
    boot_keys = ([f"{a}_minus_{b}/{mt}" for a, b in PAIRS for mt in ("full", "tail")]
                 + [f"level_{k}/{mt}" for k in LEVELS for mt in ("full", "tail")])
    SB = StratBoot(boot_keys)

    per_state = []
    arr = {}
    out = {}

    def flush():
        out["minutes"] = round((time.time() - t00) / 60, 2)
        out["per_state"] = per_state
        with open(OUT_JSON + ".tmp", "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=1, default=float)
        os.replace(OUT_JSON + ".tmp", OUT_JSON)
        savez_atomic(OUT_NPZ, **arr)

    out.update({
        "analysis": "paperC_C4B_translation_noise",
        "design_sheet": "docs/design/DS_C4B_TRANSLATION_NOISE.md (PRE-REGISTERED 2026-08-25)",
        "author_ruling": ("PAPER_C_DESIGN.md AUTHOR RULING 2026-08-25 -- declared circularity is "
                          "admitted for a RELATIVE measurement (the translation cost). Absolute rows "
                          f"are printed and labelled '{CIRC}' and are never headlined."),
        "produced_by": "src/instrument/c4b_noise.py",
        "citable": True, "test_touched": False, "split": "validation",
        "n_states": N_STATES, "arms": list(ARMS),
        "arm_definitions": {
            "L": "top-4 concepts by cos(d_c, R_s) in tower space, ties by concept id (no text)",
            "T": "SBERT top-4 (mincos 0.30) of adapter A's banked concept-hot render of the m=5 "
                 "blend target",
            "Tprime": "the same, from the scrambled twin A' (correspondence-destroying control)",
            "P": "the same, from the channel-matched prior-only text (zeroed latent + concept hot)"},
        "answer_model": dict(ans_diag, **ins),
        "declared_choices": {
            "base_is_m0_no_fold": ("the arms are scored against the m=0 (no-fold) intercept on the "
                                   "state's own cohort; the state's interview PREFIX is not folded "
                                   "in any arm (the sheet's base row). Arm-symmetric."),
            "ask_order": ("ascending concept id within an arm, so two arms selecting the SAME set "
                          "fold identically. Arm-symmetric."),
            "prior_text_arm": ("P uses adapter A's concept-hot prior-only text, the channel-matched "
                               "one, identical for all 42 states")},
        "measure": ("each arm's concepts asked as questions, answered by the declared-circular sign "
                    "rule at +-1, folded by the env's certified ConceptFoldNet(cv3) path and scored "
                    "full + tail NDCG@10 by genv_eval -- policy-B pool, learned decoder bias, mask "
                    "contract live. No second scorer exists in this file."),
        "bootstrap": {"n_resamples": N_BOOT, "seed": BOOT_SEED,
                      "per_state": "paired percentile over the state's own user-rows",
                      "summary": ("stratified: users resampled WITHIN state, states fixed, weight = "
                                  "cohort size (the pooled user-row mean)"),
                      "declared_unit": "user-rows within state, states nested",
                      "nesting_caveat": ("the 42 states are nested, so a user can appear in several; "
                                         "the summary treats states as independent strata and "
                                         "therefore UNDERSTATES its interval. The per-state rows are "
                                         "the primary object."),
                      "chunked_bootstrap_selftest": bt},
        "reference": {"definition": ("R_s = sum_j w_j * tower(q_j) over the state's banked m=5 "
                                     "ingredients (c4_pilot.reference_directions, IMPORTED)"),
                      "disclosure": r_disc},
        "render_provenance": render_prov,
        "prior_only_text": prior_txt,
        "sbert": {"model": br.RE.SBERT_MODEL, "pooling": br.RE.SBERT_MODE,
                  "topn": br.RE.SBERT_TOPN, "mincos": br.RE.SBERT_MINCOS},
        "canonical_snap": {"val_m0_full": f0, "val_m0_tail": t0_},
        "checkpoint": {"path": os.path.relpath(args.ckpt, _ROOT), "sha256": ck_sha},
        "asserts": {"pool2full_index_map_max_abs_err": map_err,
                    "target_rebuild_max_abs_diff": d_tgt,
                    "point_target_vs_banked_max_abs_diff": d_pt,
                    "R_vs_banked_max_abs_diff": d_R,
                    "concept_dirs_reproduced_worst_1_minus_abscos": br.dc_reproduction_worst,
                    "renders_json_npz_disagreements": int(n_txt_bad)},
        "what_may_not_be_claimed": (
            "no absolute 'asking it helps' claim (the ruling's fatal use stays closed); nothing "
            "about the item channel; nothing from emission renders (retracted); no generalisation "
            "beyond the 42 nested, structurally-selected states -- the C1 population caveats travel"),
        "git": GB,
        "per_state": [],
        "readings": {}})
    arr["sid"] = np.asarray([s["sid"] for s in reg], object)
    arr["overlap_L_inter_T"] = overlap_LT
    arr["R"] = R
    arr["cos_tower"] = cos_tower.astype(np.float32)
    flush()

    log("#" * 110)
    log(f"# scoring {N_STATES} states x ({len(ARMS)} arms + the m=0 base) through genv_eval")
    for i in range(N_STATES):
        s = reg[i]
        rws = [int(r) for r in s["rows"]]
        vals = {}
        cells = {}
        bf, bt_ = genv_eval(E, [], rws)
        assert np.isfinite(bf).all(), "a base full-NDCG is NaN -- a cohort row has no target"
        tmask = np.isfinite(bt_)
        vals["level_base/full"] = bf
        vals["level_base/tail"] = bt_[tmask]
        cells["base"] = {"full": boot_mean_chunked(bf), "tail": boot_mean_chunked(bt_)}
        arr[f"s{i}_rows"] = np.asarray(rws, np.int64)
        arr[f"s{i}_base_full"] = bf.astype(np.float32)
        arr[f"s{i}_base_tail"] = bt_.astype(np.float32)
        F, T = {}, {}
        for a in ARMS:
            seq = [(1, int(c)) for c in arm_c[a][i]]                # ascending concept id
            f, t = genv_eval(E, seq, rws)
            assert np.isfinite(f).all()
            assert (np.isfinite(t) == tmask).all(), "the tail-eligible set moved between arms"
            F[a], T[a] = f, t
            vals[f"level_{a}/full"] = f
            vals[f"level_{a}/tail"] = t[tmask]
            cells[a] = {"full": boot_mean_chunked(f), "tail": boot_mean_chunked(t),
                        "n_concepts": len(arm_c[a][i]), "concepts": arm_meta[a][i]}
            arr[f"s{i}_{a}_full"] = f.astype(np.float32)
            arr[f"s{i}_{a}_tail"] = t.astype(np.float32)
        diffs = {}
        for x, y in PAIRS:
            k = f"{x}_minus_{y}"
            vals[f"{k}/full"] = F[x] - F[y]
            vals[f"{k}/tail"] = (T[x] - T[y])[tmask]
            diffs[k] = {"full": boot_mean_chunked(F[x] - F[y]),
                        "tail": boot_mean_chunked(T[x] - T[y])}
            diffs[k]["full"]["directional_pass"] = directional(diffs[k]["full"])
            diffs[k]["tail"]["directional_pass"] = directional(diffs[k]["tail"])
        SB.add_state(vals)
        steps, ok_path = POC.path_in_words(s, br.names, ni, pool2full)
        per_state.append({
            "i": i, "sid": s["sid"], "group": s["group"], "depth": len(s["prefix"]),
            "n_users": len(rws), "n_tail_eligible": int(tmask.sum()),
            "sid_path_matches_prefix": bool(ok_path), "interview_path": steps,
            "render_A": txt_A[i], "render_Aprime": txt_AP[i],
            "overlap_L_inter_T": int(overlap_LT[i]),
            "arm_concepts": {a: [x["concept"] for x in arm_meta[a][i]] for a in ARMS},
            "levels_" + CIRC.replace(" ", "_"): cells,
            "differences": diffs})
        flush()
        c = diffs["L_minus_T"]
        log(f"  [{i + 1:>2}/{N_STATES}] {s['sid']:<32} n={len(rws):>5} |LnT|={overlap_LT[i]} "
            f"render={txt_A[i]!r:<18} L-T full {c['full']['mean']:+.5f} "
            f"[{c['full']['ci_lo']:+.5f}, {c['full']['ci_hi']:+.5f}] tail "
            f"{c['tail']['mean']:+.5f} [{c['tail']['ci_lo']:+.5f}, {c['tail']['ci_hi']:+.5f}] "
            f"({(time.time() - t00) / 60:.1f}m)")

    # ---- the registered readings --------------------------------------------------------------------
    summary = {k: SB.read(k) for k in boot_keys}
    for k, v in summary.items():
        if "_minus_" in k:
            v["directional_pass"] = directional(v)
    n_pass = {f"{x}_minus_{y}/{mt}":
              int(sum(1 for r in per_state
                      if r["differences"][f"{x}_minus_{y}"][mt]["directional_pass"]))
              for x, y in PAIRS for mt in ("full", "tail")}
    n_neg = {f"{x}_minus_{y}/{mt}":
             int(sum(1 for r in per_state
                     if (r["differences"][f"{x}_minus_{y}"][mt]["ci_hi"] or 0.0) < 0))
             for x, y in PAIRS for mt in ("full", "tail")}
    ov_hist = {str(int(k)): int(v) for k, v in
               zip(*np.unique(overlap_LT, return_counts=True))}
    out["readings"] = {
        "reading1_translation_cost_L_minus_T": {
            "what": "the headline: the COST of saying the direction in English, reported as a cost "
                    "whichever sign it takes",
            "summary_cohort_weighted": {mt: summary[f"L_minus_T/{mt}"] for mt in ("full", "tail")},
            "n_states_CI_above_zero": {mt: n_pass[f"L_minus_T/{mt}"] for mt in ("full", "tail")},
            "n_states_CI_below_zero": {mt: n_neg[f"L_minus_T/{mt}"] for mt in ("full", "tail")}},
        "reading2_T_minus_Tprime": {
            "what": "the true render must beat the scrambled twin's render through the same "
                    "simulator (DIRECTIONAL: CI must exclude zero to pass)",
            "summary_cohort_weighted": {mt: summary[f"T_minus_Tprime/{mt}"] for mt in
                                        ("full", "tail")},
            "verdict": {mt: ("PASS" if summary[f"T_minus_Tprime/{mt}"]["directional_pass"]
                             else "FAIL") for mt in ("full", "tail")},
            "n_states_CI_above_zero": {mt: n_pass[f"T_minus_Tprime/{mt}"] for mt in
                                       ("full", "tail")},
            "n_states_CI_below_zero": {mt: n_neg[f"T_minus_Tprime/{mt}"] for mt in
                                       ("full", "tail")}},
        "reading3_T_minus_P": {
            "what": "the render must beat channel prior text (DIRECTIONAL)",
            "summary_cohort_weighted": {mt: summary[f"T_minus_P/{mt}"] for mt in ("full", "tail")},
            "verdict": {mt: ("PASS" if summary[f"T_minus_P/{mt}"]["directional_pass"] else "FAIL")
                        for mt in ("full", "tail")},
            "n_states_CI_above_zero": {mt: n_pass[f"T_minus_P/{mt}"] for mt in ("full", "tail")},
            "n_states_CI_below_zero": {mt: n_neg[f"T_minus_P/{mt}"] for mt in ("full", "tail")}},
        "reading4_overlap_L_inter_T": {
            "what": "|concepts(L) n concepts(T)| per state, 0-4, printed beside every row",
            "histogram": ov_hist, "distribution": dist(overlap_LT),
            "note": ("if T ~ L because the SETS coincide, that is the finding, not a confound -- the "
                     "ask order is ascending concept id in both arms, so a coincident set folds "
                     "identically and its difference is exactly zero")},
        "reading5_absolute_levels_" + CIRC.replace(" ", "_"): {
            "label": CIRC,
            "cohort_weighted": {k: {mt: summary[f"level_{k}/{mt}"] for mt in ("full", "tail")}
                                for k in LEVELS},
            "note": ("PRINTED, never headlined, never compared to any non-simulated number "
                     "(the AUTHOR RULING 2026-08-25 keeps the absolute use closed)")}}
    out["summary_all_cells"] = summary
    flush()

    # ---- the console read ---------------------------------------------------------------------------
    log("=" * 110)
    log(f"READING 1 -- TRANSLATION COST  L - T  (cohort-weighted; unit = user-rows within state, "
        f"states nested)")
    for mt in ("full", "tail"):
        v = summary[f"L_minus_T/{mt}"]
        log(f"  {mt:<5} {v['mean']:+.6f}  CI [{v['ci_lo']:+.6f}, {v['ci_hi']:+.6f}]  "
            f"n_user_rows={v['n_user_rows']:,}  states CI>0 {n_pass[f'L_minus_T/{mt}']}/42, "
            f"CI<0 {n_neg[f'L_minus_T/{mt}']}/42")
    for nm, ttl in (("T_minus_Tprime", "READING 2 -- T - T' (scrambled twin)"),
                    ("T_minus_P", "READING 3 -- T - P (prior text)")):
        log(f"{ttl}  [directional: CI must exclude zero]")
        for mt in ("full", "tail"):
            v = summary[f"{nm}/{mt}"]
            log(f"  {mt:<5} {v['mean']:+.6f}  CI [{v['ci_lo']:+.6f}, {v['ci_hi']:+.6f}]  -> "
                f"{'PASS' if v['directional_pass'] else 'FAIL'}   states CI>0 "
                f"{n_pass[f'{nm}/{mt}']}/42, CI<0 {n_neg[f'{nm}/{mt}']}/42")
    log(f"READING 4 -- |concepts(L) n concepts(T)| histogram over the 42 states: {ov_hist} "
        f"(mean {overlap_LT.mean():.3f})")
    log(f"READING 5 -- ABSOLUTE LEVELS, {CIRC.upper()} (printed, never headlined)")
    for k in LEVELS:
        a, b = summary[f"level_{k}/full"], summary[f"level_{k}/tail"]
        log(f"  {k:<8} full {a['mean']:.5f} [{a['ci_lo']:.5f}, {a['ci_hi']:.5f}] | "
            f"tail {b['mean']:.5f} [{b['ci_lo']:.5f}, {b['ci_hi']:.5f}]")
    log("=" * 110)
    log(f"{'sid':<32} | {'render A':<18} | {'|LnT|':>5} | {'L-T full':>10} | {'T-T′ full':>10}")
    for r in per_state:
        log(f"{r['sid']:<32} | {r['render_A']:<18} | {r['overlap_L_inter_T']:>5} | "
            f"{r['differences']['L_minus_T']['full']['mean']:>+10.5f} | "
            f"{r['differences']['T_minus_Tprime']['full']['mean']:>+10.5f}")
    log("=" * 110)
    bad = [r["sid"] for r in per_state if not r["sid_path_matches_prefix"]]
    if bad:
        log(f"[WARN] {len(bad)} state(s) whose sid path length != prefix length: {bad}")
    log(f"[done] {OUT_JSON} + {os.path.basename(OUT_NPZ)}  ({out['minutes']:.1f}m)")
    return out


def main():
    ap = argparse.ArgumentParser(description="Paper C / C4B -- docs/design/DS_C4B_TRANSLATION_NOISE.md")
    ap.add_argument("--ckpt", default=E1_CKPT)
    ap.add_argument("--snapshot", default=None)
    ap.add_argument("--concept_ckpt", default=None)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--allow_dirty", action="store_true",
                    help="EMERGENCY ONLY: run with an uncommitted src/instrument (HARD RULE 10)")
    args = ap.parse_args()
    args.threads = min(int(args.threads), 4)
    from strategy_channel_suite import CERT_TOWER, CERT_CONC
    args.snapshot = args.snapshot or CERT_TOWER
    args.concept_ckpt = args.concept_ckpt or CERT_CONC
    torch.set_num_threads(int(args.threads))
    log("#" * 110)
    log("# PAPER C / C4B -- TRANSLATION NOISE. The answer model is DECLARED CIRCULAR and carries a "
        "RELATIVE claim only (the translation cost L - T). VALIDATION ONLY; the 10,000 TEST users "
        "are not built, not loaded, not touched. No LLM API.")
    log("#" * 110)
    return run(args)


if __name__ == "__main__":
    main()
