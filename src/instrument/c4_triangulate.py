r"""PAPER C / C4 PILOT -- TRIANGULATION: do the concept-channel renders sit among the point's OWN
nearest bank concepts?

AUTHOR'S QUESTION (light analysis, 2026-08-25): the C4 pilot's concept-hot renders of the m=5 BLEND
TARGETS are single English words ("religion", "horror", "science", "animals"). Reading 2 says they
are point-specific and reading 3 says they carry the direction the nearest catalogue text does not --
but neither says the word makes SEMANTIC sense for that state. This file triangulates the word
against the point's own neighbourhood, in the two spaces that already exist:

  SPACE 1 (the emitter's own head)  the top-10 CONCEPT bank entries nearest the blend target in
                                    MEAN-REMOVED k(phi) -- the same convention every C3/C3B/C4
                                    cosine uses, with the bank mean `mu` READ from the C3B pairs
                                    file, never recomputed.
  SPACE 2 (the independent tower)   the top-10 concepts by cos(d_c, R_s), where R_s is rebuilt by
                                    `c4_pilot.reference_directions` (IMPORTED, not reimplemented)
                                    and asserted bit-identical to the R banked by the pilot.

The render's OWN concept is the exact genome tag when the render IS a tag (normalised string match
against the SBERT tag column, which the bridge already asserts equals the concept channel), else the
render's top SBERT concept -- i.e. the first entry of the fold path's own top-4 selection. The three
registered-in-the-brief scores, over the 42 states:

  (a) HIT RATE   is the render's own concept inside the point's top-10, per space and in either;
                 Wilson 95% interval.
  (b) RANK       the render's own concept's rank in the FULL 1,031-concept ranking by cos to R_s;
                 median + IQR.
  (c) FLOOR      the same two statistics for ONE RANDOM TAG per state (seeded 20260825).

SCOPE. Concept-hot renders of the BLEND TARGETS under adapter A only -- the cell that passed. The
emission cells are RETRACTED (C4_PILOT.md) and are not read here. This is a QUALITATIVE / descriptive
triangulation: no NDCG, no answer simulated, no test user built, loaded or touched, no LLM API. The
renders themselves are READ from `c4_pilot_per_state.npz`, not regenerated, so the words analysed are
exactly the banked ones.
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

import re
import json
import time
import math
import argparse
import numpy as np
import torch

from policy_net import Policy, build_phi, load_traj, log, SEED
from c0_geometry import (PAPERC, E1_CKPT, N_BANK, VAL_M0_FULL, sha256, git_block, bank_vectors,
                         index_map_assert)
import c1_pilot as C1
import c3_poc_extract as POC
import c3b_adapter as C3B
import c4_pilot as C4

OUT_JSON = os.path.join(PAPERC, "c4_triangulate.json")
IN_NPZ = os.path.join(PAPERC, "c4_pilot_per_state.npz")

M = C4.M
N_STATES = C4.N_STATES
TOPK = 10                       # the neighbourhood size the brief fixes
RANDOM_SEED = 20260825          # DECLARED: the random-tag floor, and nothing else
CELL = ("target", "concept_hot", "A")     # the cell that passed; emissions are retracted


def norm_txt(s):
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def wilson(k, n, z=1.96):
    """Wilson score interval for a binomial proportion (small n, so never the normal approximation)."""
    if n == 0:
        return {"k": 0, "n": 0, "p": None, "lo": None, "hi": None}
    p = k / n
    d = 1.0 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return {"k": int(k), "n": int(n), "p": float(p), "lo": float(max(0.0, c - h)),
            "hi": float(min(1.0, c + h))}


def quartiles(v):
    v = np.asarray(v, np.float64)
    v = v[np.isfinite(v)]
    if not len(v):
        return {"n": 0, "median": None, "q1": None, "q3": None, "iqr": None,
                "min": None, "max": None, "mean": None}
    q1, med, q3 = (float(x) for x in np.percentile(v, [25, 50, 75]))
    return {"n": int(len(v)), "median": med, "q1": q1, "q3": q3, "iqr": q3 - q1,
            "min": float(v.min()), "max": float(v.max()), "mean": float(v.mean())}


def run(args):
    from ndcg_tree import build_pool
    from strategy_channel_suite import genv_mean
    t00 = time.time()
    os.makedirs(PAPERC, exist_ok=True)
    GB = git_block(allow_dirty=args.allow_dirty, threads=args.threads)      # HARD RULE 10, first
    log(f"[hardrule10] src/instrument clean={GB['src_instrument_clean']} rev={GB['rev'][:8]} "
        f"threads={GB['threads']} torch_threads={GB['torch_num_threads']}")

    # ---- the bridge (env + tower targets + SBERT), exactly as the pilot builds it ----------------
    br = C3B.TowerBridge(args)
    E, ni, C = br.E, br.ni, br.C
    blob = torch.load(args.ckpt, map_location="cpu")
    ck_sha = sha256(args.ckpt)
    assert ck_sha == C1.C0_CKPT_SHA == br.ckpt_sha

    f0, t0_ = genv_mean(E, [], E["rows"])                    # HARD RULE 5 canonical snap
    assert round(f0, 8) == round(VAL_M0_FULL, 8), f"canonical snap FAILED: {f0!r}"
    log(f"[snap] validation m=0 intercept full={f0!r} tail={t0_!r} -- canonical")

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

    # ---- the adapter's space constants: READ from the C3B pairs, never recomputed ----------------
    zp = np.load(C3B.PAIRS_NPZ)
    mu = np.asarray(zp["mu"], np.float64)
    Kbank = np.asarray(zp["K"], np.float64)
    assert Kbank.shape == (N_BANK, Kpool.shape[1])
    map_err = index_map_assert(Kbank, pool2full, Kpool.astype(np.float64))
    Kc_conc = Kbank[ni:] - mu                       # the 1,031 concept rows, MEAN-REMOVED
    assert Kc_conc.shape[0] == C
    log(f"[space] k(phi) mean-removed; concept block {Kc_conc.shape}; pool2full exact "
        f"(max|diff| {map_err:g})")

    # ---- the 42 states and their m=5 blends -------------------------------------------------------
    Dva = load_traj(train=False)
    reg, extra = C1.load_the_42(pool_index, Dva)
    states = reg + extra
    assert len(reg) == N_STATES
    for s in states:
        s["pool_prefix_idx"] = [pool_index[q] for q in s["prefix"]]
    Tt, TKt, meta = C1.build_targets(states, Epool, Kpool, pool_index, M)

    npz = np.load(IN_NPZ, allow_pickle=True)
    d = float(np.abs(Tt - np.load(os.path.join(PAPERC, "c1_pilot_per_state.npz"))
                     [f"target_m{M}"]).max())
    assert d == 0.0, f"the rebuilt m={M} targets are not the pilot's (max|diff| {d:g})"
    d_pt = float(np.abs(TKt[:N_STATES].astype(np.float32)
                        - np.asarray(npz["point_target"], np.float32)).max())
    assert d_pt == 0.0, f"the rebuilt target points are not the C4 pilot's (max|diff| {d_pt:g})"

    # ---- R_s: IMPORTED from the pilot, then asserted against the banked R -------------------------
    R, r_disc = C4.reference_directions(br, meta, pool, pool2full)
    d_R = float(np.abs(R - np.asarray(npz["R"], np.float64)).max())
    assert d_R == 0.0, f"the rebuilt R_s is not the pilot's banked R (max|diff| {d_R:g})"
    log(f"[reference] R_s {R.shape} rebuilt bit-identically to the pilot's banked R")

    # ---- the two neighbourhoods ---------------------------------------------------------------------
    Pt = TKt[:N_STATES].astype(np.float64) - mu
    K_unit = Kc_conc / np.linalg.norm(Kc_conc, axis=1, keepdims=True)
    P_unit = Pt / np.linalg.norm(Pt, axis=1, keepdims=True)
    cos_kphi = P_unit @ K_unit.T                                  # (42, 1031)
    D_unit = br.d_c / np.linalg.norm(br.d_c, axis=1, keepdims=True)
    R_unit = R / np.linalg.norm(R, axis=1, keepdims=True)
    cos_tower = R_unit @ D_unit.T                                 # (42, 1031)
    assert cos_kphi.shape == cos_tower.shape == (N_STATES, C)

    ord_kphi = np.argsort(-cos_kphi, axis=1)
    ord_tower = np.argsort(-cos_tower, axis=1)
    rank_tower = np.empty_like(ord_tower)                         # rank 1 = nearest to R_s
    rank_kphi = np.empty_like(ord_kphi)
    for i in range(N_STATES):
        rank_tower[i, ord_tower[i]] = np.arange(1, C + 1)
        rank_kphi[i, ord_kphi[i]] = np.arange(1, C + 1)

    # ---- the renders (READ, not regenerated) and their own concepts ------------------------------
    texts = [str(x) for x in npz["text_" + "_".join(CELL)]]
    banked_cos = np.asarray(npz["cos_" + "_".join(CELL)], np.float64)
    assert len(texts) == N_STATES
    sel = br.select(texts)                                        # the fold path's own top-4
    tagn = [norm_txt(t) for t in br.tag_names]
    tag_lookup = {}
    for c, t in enumerate(tagn):
        tag_lookup.setdefault(t, c)                    # first wins; collisions counted, not hidden
    n_tag_collisions = C - len(tag_lookup)
    if n_tag_collisions:
        log(f"[tags] {n_tag_collisions} genome tag name(s) collide after normalisation -- the "
            f"lowest concept id wins the exact-match rule (declared, counted in the JSON)")

    own, own_how = [], []
    for i, tx in enumerate(texts):
        n = norm_txt(tx)
        if n in tag_lookup:
            own.append(tag_lookup[n]); own_how.append("exact_tag")
        elif sel[i]:
            own.append(int(sel[i][0][0])); own_how.append("top_sbert")
        else:
            own.append(-1); own_how.append("non_encodable")
    own = np.asarray(own, np.int64)
    n_enc = int((own >= 0).sum())
    log(f"[render] {len(set(texts))} distinct texts; own concept resolved for {n_enc}/{N_STATES} "
        f"(exact tag {own_how.count('exact_tag')}, top SBERT {own_how.count('top_sbert')}, "
        f"non-encodable {own_how.count('non_encodable')})")

    # ---- the random-tag floor ----------------------------------------------------------------------
    rng = np.random.default_rng(RANDOM_SEED)
    rand = rng.integers(0, C, size=N_STATES).astype(np.int64)

    # ---- the three scores ----------------------------------------------------------------------------
    def scores(cid):
        hk = ht = he = 0
        rt, rk = [], []
        n = 0
        for i in range(N_STATES):
            c = int(cid[i])
            if c < 0:
                continue
            n += 1
            a = bool(rank_kphi[i, c] <= TOPK)
            b = bool(rank_tower[i, c] <= TOPK)
            hk += a; ht += b; he += (a or b)
            rt.append(int(rank_tower[i, c])); rk.append(int(rank_kphi[i, c]))
        return {"n_scored": n,
                "hit_top10_kphi": wilson(hk, n),
                "hit_top10_tower": wilson(ht, n),
                "hit_top10_either": wilson(he, n),
                "rank_in_tower_ranking": quartiles(rt),
                "rank_in_kphi_ranking": quartiles(rk)}

    sc_render = scores(own)
    sc_random = scores(rand)

    # ---- the qualitative table -----------------------------------------------------------------------
    rows, n_path_bad = [], []
    for i in range(N_STATES):
        s = reg[i]
        steps, ok = POC.path_in_words(s, br.names, ni, pool2full)
        if not ok:
            n_path_bad.append(s["sid"])
        ing = []
        for j, qi in enumerate(meta[i]["top_m_pool_idx"]):
            b = int(pool2full[int(qi)])
            ing.append({"bank_idx": b, "channel": "concept" if b >= ni else "item",
                        "name": POC.phrase(br.names, ni, b),
                        "weight": float(meta[i]["weights"][j])})
        c = int(own[i])
        rows.append({
            "i": i, "sid": s["sid"], "group": s["group"], "depth": len(s["prefix"]),
            "n_users": len(s["rows"]),
            "interview_path": steps,
            "render": texts[i],
            "render_cos_to_R_banked": float(banked_cos[i]) if np.isfinite(banked_cos[i]) else None,
            "render_own_concept": None if c < 0 else br.tag_names[c],
            "render_own_concept_how": own_how[i],
            "render_own_concept_rank_tower": None if c < 0 else int(rank_tower[i, c]),
            "render_own_concept_rank_kphi": None if c < 0 else int(rank_kphi[i, c]),
            "render_own_concept_cos_tower": None if c < 0 else float(cos_tower[i, c]),
            "render_sbert_top4": [{"concept": br.tag_names[int(a)], "sbert_cos": float(b)}
                                  for a, b in sel[i]],
            "top10_tower": [{"rank": r + 1, "concept": br.tag_names[int(cc)],
                             "cos_to_R": float(cos_tower[i, cc])}
                            for r, cc in enumerate(ord_tower[i, :TOPK])],
            "top10_kphi": [{"rank": r + 1, "concept": br.tag_names[int(cc)],
                            "cos_meanremoved": float(cos_kphi[i, cc])}
                           for r, cc in enumerate(ord_kphi[i, :TOPK])],
            "blend_ingredients": ing,
            "random_floor_tag": br.tag_names[int(rand[i])],
            "random_floor_rank_tower": int(rank_tower[i, int(rand[i])]),
            "random_floor_rank_kphi": int(rank_kphi[i, int(rand[i])])})

    out = {
        "analysis": "paperC_C4_pilot_triangulation",
        "question": ("do the C4 pilot's concept-channel renders of the m=5 blend targets correspond "
                     "to the point's OWN nearest concept-bank entries, in the emitter's head space "
                     "(mean-removed k(phi)) and in the independent tower space (cos to R_s)?"),
        "produced_by": "src/instrument/c4_triangulate.py",
        "reads": {"renders": os.path.relpath(IN_NPZ, _ROOT), "cell": "/".join(CELL),
                  "note": ("the renders are READ from the pilot's npz, never regenerated; the "
                           "EMISSION cells are retracted (docs/results/C4_PILOT.md) and unread")},
        "citable": True, "test_touched": False, "split": "validation",
        "answer_free": ("descriptive triangulation only: no NDCG, no answer simulated to any "
                        "rendered text, no claim about the value of asking"),
        "n_states": N_STATES, "n_concepts": C, "topk": TOPK,
        "own_concept_rule": ("exact genome-tag match on the normalised render if the render IS a "
                             "tag, else the render's TOP SBERT concept (the first of the fold "
                             "path's own top-4 at mincos 0.30)"),
        "n_normalised_tag_name_collisions": int(n_tag_collisions),
        "own_concept_resolution": {"exact_tag": own_how.count("exact_tag"),
                                   "top_sbert": own_how.count("top_sbert"),
                                   "non_encodable": own_how.count("non_encodable")},
        "render_scores": sc_render,
        "random_tag_floor": sc_random,
        "floor_seed": RANDOM_SEED,
        "spaces": {
            "kphi": ("top-10 CONCEPT bank entries nearest the blend target in MEAN-REMOVED k(phi); "
                     "mu READ from the C3B pairs file"),
            "tower": ("top-10 concepts by cos(d_c, R_s), R_s rebuilt by "
                      "c4_pilot.reference_directions (IMPORTED) and asserted == the banked R")},
        "reference_disclosure": r_disc,
        "table": rows,
        "asserts": {"pool2full_index_map_max_abs_err": map_err,
                    "target_rebuild_max_abs_diff": d,
                    "point_target_vs_banked_max_abs_diff": d_pt,
                    "R_vs_banked_max_abs_diff": d_R,
                    "concept_dirs_reproduced_worst_1_minus_abscos": br.dc_reproduction_worst,
                    "states_whose_sid_path_did_not_match_the_prefix": n_path_bad},
        "canonical_snap": {"val_m0_full": f0, "val_m0_tail": t0_},
        "checkpoint": {"path": os.path.relpath(args.ckpt, _ROOT), "sha256": ck_sha},
        "sbert": {"model": br.RE.SBERT_MODEL, "pooling": br.RE.SBERT_MODE,
                  "topn": br.RE.SBERT_TOPN, "mincos": br.RE.SBERT_MINCOS},
        "what_may_not_be_claimed": ("no NDCG, no 'asking it would help', no human-plausibility "
                                    "grading; n = 42 nested, structurally-selected states (the C1 "
                                    "caveats travel); the k(phi) neighbourhood is an anisotropic "
                                    "space with a documented common mode"),
        "git": GB,
        "minutes": round((time.time() - t00) / 60, 2)}
    with open(OUT_JSON + ".tmp", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, default=float)
    os.replace(OUT_JSON + ".tmp", OUT_JSON)

    # ---- the console read ---------------------------------------------------------------------------
    log("=" * 110)
    log(f"TRIANGULATION -- render's own concept vs the point's top-{TOPK} neighbourhood "
        f"(n={sc_render['n_scored']})")
    for k in ("hit_top10_kphi", "hit_top10_tower", "hit_top10_either"):
        a, b = sc_render[k], sc_random[k]
        log(f"  {k:<18} render {a['k']:>2}/{a['n']} = {a['p']:.4f} "
            f"[{a['lo']:.4f}, {a['hi']:.4f}]   |   random tag {b['k']:>2}/{b['n']} = {b['p']:.4f} "
            f"[{b['lo']:.4f}, {b['hi']:.4f}]")
    for k in ("rank_in_tower_ranking", "rank_in_kphi_ranking"):
        a, b = sc_render[k], sc_random[k]
        log(f"  {k:<22} render median {a['median']:.1f} IQR [{a['q1']:.1f}, {a['q3']:.1f}]   |   "
            f"random median {b['median']:.1f} IQR [{b['q1']:.1f}, {b['q3']:.1f}]  (of {C})")
    log("=" * 110)
    log(f"{'sid':<34} | {'render':<18} | top-5 tower neighbours")
    for r in rows:
        log(f"{r['sid']:<34} | {r['render']:<18} | "
            + ", ".join(x["concept"] for x in r["top10_tower"][:5]))
    log("=" * 110)
    log(f"[done] {OUT_JSON} ({out['minutes']:.1f}m)")
    if n_path_bad:
        log(f"[WARN] {len(n_path_bad)} state(s) whose sid path length != prefix length: {n_path_bad}")
    return out


def main():
    ap = argparse.ArgumentParser(description="Paper C / C4 pilot -- concept-render triangulation")
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
    log("# PAPER C / C4 PILOT TRIANGULATION -- do the rendered words sit among the point's own "
        "nearest concepts? Descriptive; no NDCG, no answer simulated, no test user touched.")
    log("#" * 110)
    return run(args)


if __name__ == "__main__":
    main()
