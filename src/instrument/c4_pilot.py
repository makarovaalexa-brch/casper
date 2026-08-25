r"""PAPER C / STAGE C4 -- PILOT: render the policy's OWN emitted points, fold them back, measure
faithfulness.

Design sheet: `docs/design/DS_C4_PILOT.md` (PRE-REGISTERED 2026-08-25, author approved, BINDING).
Read it first. This file closes the loop  policy -> point -> English -> fold  and is ANSWER-FREE
throughout: no answer is simulated to any rendered question, no NDCG is computed anywhere, and no
claim about "asking it would help" can be built from these outputs (the simulator-circularity ruling
stays closed). What is measured is whether the TEXT is a faithful carrier of the POINT.

THE OBJECTS
-----------
  points      the C1 pilot's registered 42 states, two vectors each -- the m=5 blend TARGET and the
              full-policy/seed-1 EMISSION -- in the bias-free space k(phi), read from
              `c1_pilot_per_state.npz` under the SAME provenance gate `c3_poc_extract` uses (the
              targets are rebuilt bit-identically through `c1_pilot.build_targets`; the emission
              array is pinned to the C1 commit and its two banked criterion medians). Plus 42
              SPECTRUM-MATCHED RANDOM points (the repaired MEAN-REMOVED-space rescale, seeded) as
              the point-level null.
  renderers   adapter A (`c3b_true_best.pt`) and its scrambled twin A' (`c3b_scrambled_best.pt`),
              both split-hash asserted against the C3B manifest. A POLICY POINT HAS NO CHANNEL, so
              every point is rendered under BOTH channel one-hots and both rows are reported in
              full -- never pooled, no post-hoc primary. Prior-only text per channel comes from the
              stage-2 construction (zeroed latent + channel one-hot).
  reference   R_s = sum_j w_j * tower(q_j) over the state's OWN m=5 measured ingredients, with
              `tower(.)` the C3B bridge's target map (item -> centred, popularity-stripped decoder
              row `Ew`; concept -> `build_concept_dirs` direction `d_c`). R_s is the reference for
              BOTH the target and the emission at that state, and for the random point standing in
              for it (declared in the sheet).
  measure     rendered text -> `TowerBridge` (SBERT -> concepts -> cv3 fold displacement,
              popularity-stripped) -> cos(u_text, R_s). THE FOLD IS THE FOLD-BACK.

WHAT IS IMPORTED, NEVER REIMPLEMENTED
-------------------------------------
  `c3b_adapter`      TowerBridge, load_adapter, greedy, input_vectors, zero_latent_input, rowcos,
                     boot_mean, ensure_lm/load_lm, and the manifest constants for mean-removal and
                     the global scale (read from `c3b_pairs.npz`, never recomputed here);
  `c1_pilot`         load_the_42, build_targets, the C0 checkpoint sha;
  `c3_poc_extract`   the C1 emission provenance pins, `phrase`, `path_in_words`;
  `c0_geometry`      bank_vectors, index_map_assert, git_block, savez_atomic.

THE FOUR REGISTERED READINGS (directional, as the sheet specifies; paired percentile bootstrap over
the 42 states, 10,000 resamples, seed 98765, per point-kind x channel-hot cell)
  1  FAITHFULNESS (the carrying criterion)   A(point) - A'(point) > 0, CI excluding zero.
  2  POINT-SPECIFICITY                        A(real point) - A(random point) > 0, CI excluding
                                              zero. Registered as UNCERTAIN: a null is reportable.
  3  CATALOGUE COMPARATOR, REPORT-ONLY, NO BAR   the state's nearest bank question's GOLD phrase
                                              through the same bridge vs R_s.
  4  PRIOR FLOOR                              prior-only text vs R_s, must sit at ~0.
  +  the DEMO TABLE: every rendered text verbatim with the state's sid and interview path in words.

WHAT MAY NOT BE CLAIMED. No NDCG. No "asking it would help". No answer simulated to any rendered
text. No human-plausibility grading. No pooling across the declared cells. No post-hoc primary.
n = 42 nested, structurally-selected states (the C1 caveats travel). VALIDATION ONLY: the 10,000
canonical TEST users are not built, not loaded, not touched. No LLM API.
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
from c0_geometry import (PAPERC, E1_CKPT, N_BANK, VAL_M0_FULL, sha256, git, git_block,
                         savez_atomic, index_map_assert, bank_vectors)
import c1_pilot as C1
import c3_poc_extract as POC
import c3b_adapter as C3B

OUT_JSON = os.path.join(PAPERC, "c4_pilot.json")
OUT_NPZ = os.path.join(PAPERC, "c4_pilot_per_state.npz")

M = 5                                # the pilot's PRIMARY blend width
N_STATES = 42                        # the registered population
EMISSION_RUN = "zbar_full_seed1"     # the sheet's "full-policy arm, seed 1"
RANDOM_SEED = 20260825               # DECLARED: the spectrum-matched draws, and nothing else
BOOT_SEED = SEED                     # 98765, the registered bootstrap seed
KINDS = ("target", "emission", "random")
REAL_KINDS = ("target", "emission")
CHANNELS = ((0, "item_hot"), (1, "concept_hot"))
ARMS = ("A", "Aprime")


# ============================================================================ provenance
def c1_provenance():
    """`c3_poc_extract`'s C1 gate, verbatim through its own pinned constants.

    The emission array is a 42x129 float32 block under a key; that is not by itself evidence it is
    the pilot's seed-1 full-policy emission. The pins tie it to the C1 commit and to the two
    criterion numbers C1_PILOT.md reports, and the npz's OWN per-state cosines must reproduce the
    json's median -- so the two artifacts have to agree with each other as well as with the
    constants. Imported rather than re-typed: if the pins ever move, they move in ONE file.
    """
    with open(POC.C1_JSON, encoding="utf-8") as fh:
        c1j = json.load(fh)
    c1r = c1j["stage1"]["runs"]["full/seed1"]
    med_json = float(c1r["criterion1_cos_target"]["median"])
    assert c1j["git"]["rev"] in POC.C1_ACCEPTED_REVS, (
        f"c1_pilot.json was produced at {c1j['git']['rev']}, not one of the accepted C1 commits "
        f"{POC.C1_ACCEPTED_REVS} -- admit a new rev deliberately, after checking the criteria")
    assert c1j["checkpoint"]["sha256"] == C1.C0_CKPT_SHA, "c1_pilot.json is on another checkpoint"
    assert abs(med_json - POC.C1_FULL_SEED1_MEDIAN_COS) < 1e-12
    assert abs(float(c1r["criterion1_cos_target"]["mean"]) - POC.C1_FULL_SEED1_MEAN_COS) < 1e-12
    assert abs(float(c1r["criterion2_same_nearest_bank_question"])
               - POC.C1_FULL_SEED1_SAME_NEAREST) < 1e-12
    return c1j, {"c1_json": os.path.relpath(POC.C1_JSON, _ROOT),
                 "c1_git_rev": c1j["git"]["rev"], "c1_git_rev_pinned": POC.C1_GIT_REV,
                 "c1_checkpoint_sha256": c1j["checkpoint"]["sha256"],
                 "full_seed1_median_cos_json": med_json,
                 "full_seed1_median_cos_pinned": POC.C1_FULL_SEED1_MEDIAN_COS,
                 "full_seed1_mean_cos_json": float(c1r["criterion1_cos_target"]["mean"]),
                 "full_seed1_same_nearest_json":
                     float(c1r["criterion2_same_nearest_bank_question"])}


# ============================================================================ the reference R_s
def reference_directions(br, meta, pool, pool2full, n_states=N_STATES):
    """R_s = sum_j w_j * tower(q_j) over the state's OWN banked m=5 ingredients.

    `tower(.)` is `TowerBridge.target_tower` -- the SAME map the C3B gate scores its held-out
    questions against, applied here to the blend's ingredients. Every ingredient must resolve: a
    pool index maps to a (channel, id) pair, and that pair to a bank row; an unresolved one is an
    assertion, never a dropped term (a partially-built reference is indistinguishable from a
    differently-weighted one -- memory `cross-population-index-match-is-silent`).

    DISCLOSURE, not a deviation: an item's tower vector `Ew[i]` carries its own magnitude while a
    concept's `d_c[c]` is unit length, so in a mixed blend the nominal weight w_j is not the
    effective one. The sheet defines R_s in exactly these two objects, so the construction stands
    as written; the size of the disparity is measured and reported beside every number.
    """
    R = np.zeros((n_states, br.Ew.shape[1]), np.float64)
    n_item_ing = n_conc_ing = 0
    eff_item, eff_conc = [], []
    for i in range(n_states):
        q = [int(x) for x in meta[i]["top_m_pool_idx"]]
        w = np.asarray(meta[i]["weights"], np.float64)
        assert len(q) == M and len(w) == M, f"state {i}: blend width is not m={M}"
        assert abs(float(w.sum()) - 1.0) < 1e-9, f"state {i}: weights do not sum to 1"
        bidx = []
        for qi in q:
            ch, cid = pool[qi]
            if ch == 0:
                assert 0 <= cid < br.ni, f"item ingredient {cid} outside the item bank"
                n_item_ing += 1
            else:
                assert 0 <= cid < br.C, f"concept ingredient {cid} outside the concept bank"
                n_conc_ing += 1
            b = int(pool2full[qi])
            assert 0 <= b < N_BANK
            bidx.append(b)
        V = br.target_tower(np.asarray(bidx, np.int64)).astype(np.float64)
        assert V.shape == (M, R.shape[1]) and np.isfinite(V).all(), \
            f"state {i}: an ingredient did not resolve to a finite tower vector"
        nv = np.linalg.norm(V, axis=1)
        assert (nv > 1e-12).all(), f"state {i}: a zero-length ingredient vector"
        for j, qi in enumerate(q):
            (eff_item if pool[qi][0] == 0 else eff_conc).append(float(w[j] * nv[j]))
        R[i] = (w[:, None] * V).sum(0)
    assert np.isfinite(R).all() and (np.linalg.norm(R, axis=1) > 1e-10).all(), \
        "a reference direction is degenerate"
    disc = {"n_item_ingredients": n_item_ing, "n_concept_ingredients": n_conc_ing,
            "mean_effective_weight_item": float(np.mean(eff_item)) if eff_item else None,
            "mean_effective_weight_concept": float(np.mean(eff_conc)) if eff_conc else None,
            "mean_item_tower_norm": float(np.linalg.norm(br.Ew, axis=1).mean()),
            "mean_concept_tower_norm": float(np.linalg.norm(br.d_c, axis=1).mean()),
            "note": ("Ew rows carry their own magnitude, d_c rows are unit -- so in a mixed blend "
                     "the nominal weight is not the effective one. Reported, not corrected: R_s is "
                     "defined by the sheet in exactly these two objects.")}
    return R, disc


# ============================================================================ the random points
def spectrum_matched(P_all, P_norm_ref, mu, n_draw, seed):
    """Spectrum-matched random points, with the REPAIRED mean-removed-space rescale.

    Copied in construction from `c3_poc_extract` post-repair (2026-08-24): draw in the eigenbasis of
    the real points' empirical covariance about their own mean, then match the draw's MEAN-REMOVED
    length to the real point it stands in for. The rescale must happen in the mean-removed space
    because every cosine downstream is taken after the bank mean is subtracted, and the bank mean is
    enormous relative to the spread -- matching ||v|| in the raw space says almost nothing about
    length in the space the comparison actually reads.

    DECLARED CHOICE (the sheet fixes neither): the spectrum is taken over ALL 84 real vectors
    (targets and emissions together -- they are "the real points"), and each draw is rescaled to the
    mean-removed norm of its state's TARGET. Both real norm sets are banked in the output so the
    choice is visible; the target and the emission at a state sit at cos ~ 0.99, so the two possible
    reference norms are nearly the same number, and the JSON reports how nearly.
    """
    rng = np.random.default_rng(seed)
    Pm = P_all.mean(0)
    Xc = P_all - Pm
    _U, S_, Vt_ = np.linalg.svd(Xc, full_matrices=False)
    lam = (S_ ** 2) / max(len(P_all) - 1, 1)
    G = rng.standard_normal((n_draw, len(lam)))
    Rr = Pm + (G * np.sqrt(lam)) @ Vt_
    Rc = Rr - mu
    ref = np.linalg.norm(P_norm_ref - mu, axis=1)
    Rc *= (ref / np.linalg.norm(Rc, axis=1))[:, None]
    out = Rc + mu
    spec = {"n_real_points_defining_the_spectrum": int(len(P_all)),
            "rank": int((lam > 1e-12).sum()),
            "rescale_space": "mean-removed (the 2026-08-24 repaired construction)",
            "eigenvalues_top20": [float(x) for x in lam[:20]],
            "n_draws": int(n_draw), "seed": int(seed),
            "random_norms_mean_removed": [float(x) for x in np.linalg.norm(out - mu, axis=1)],
            "reference_norms_mean_removed": [float(x) for x in ref]}
    return out, spec


# ============================================================================ the run
def run(args):
    from ndcg_tree import build_pool
    from strategy_channel_suite import genv_mean
    t00 = time.time()
    os.makedirs(PAPERC, exist_ok=True)
    GB = git_block(allow_dirty=args.allow_dirty, threads=args.threads,
                   extra={"status_short": git("status", "--short")})   # HARD RULE 10, first
    log(f"[hardrule10] src/instrument clean={GB['src_instrument_clean']} rev={GB['rev'][:8]} "
        f"threads={GB['threads']} torch_threads={GB['torch_num_threads']}")

    # ---- the bridge. It asserts the C0/C1 checkpoint sha and builds the env we then reuse ---------
    br = C3B.TowerBridge(args)
    E, ni, C = br.E, br.ni, br.C
    blob = torch.load(args.ckpt, map_location="cpu")
    ck_sha = sha256(args.ckpt)
    assert ck_sha == C1.C0_CKPT_SHA == br.ckpt_sha

    # HARD RULE 5 canonical snap: this is the certified ruler, not a lookalike env
    f0, t0_ = genv_mean(E, [], E["rows"])
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
    log(f"[bank] pool e_q {Epool.shape} / k(phi) {Kpool.shape}")

    # ---- the adapter input space: the C3B manifest constants, READ, never recomputed --------------
    man = C3B.load_manifest()
    zp = np.load(C3B.PAIRS_NPZ)
    mu = np.asarray(zp["mu"], np.float64)
    scale = float(zp["scale"])
    Kbank = np.asarray(zp["K"], np.float64)
    assert Kbank.shape == (N_BANK, Kpool.shape[1])
    d_mu = float(np.abs(Kbank.mean(0) - mu).max())
    assert d_mu < 1e-9, f"the banked mean-removal constant is not the bank mean (max|diff| {d_mu:g})"
    assert abs(float(man["space"]["global_scale_mean_centred_norm"]) - scale) < 1e-12
    assert man["checkpoint"]["sha256"] == ck_sha, "the C3B pairs were built on another checkpoint"
    # the index map, against the ADAPTER'S OWN bank: correspondence exact, values at float32 rounding
    map_err = index_map_assert(Kbank, pool2full, Kpool.astype(np.float64))
    log(f"[space] mean-removal ||mu||={np.linalg.norm(mu):.4f}, global scale {scale:.4f}; "
        f"pool2full exact against the C3B bank (max|diff| {map_err:g})")

    # ---- the 42 states, the targets (bit-identity gate) and the emissions -------------------------
    Dva = load_traj(train=False)
    reg, extra = C1.load_the_42(pool_index, Dva)
    states = reg + extra
    assert len(reg) == N_STATES
    for s in states:
        s["pool_prefix_idx"] = [pool_index[q] for q in s["prefix"]]

    npz = np.load(os.path.join(PAPERC, "c1_pilot_per_state.npz"))
    Tt, TKt, meta = C1.build_targets(states, Epool, Kpool, pool_index, M)
    d = float(np.abs(Tt - npz[f"target_m{M}"]).max())
    assert d == 0.0, f"the rebuilt m={M} targets are not the pilot's (max|diff| {d:g})"
    d_bf = float(np.abs(Tt[:, :-1] - TKt).max())
    assert d_bf == 0.0, f"target[:, :-1] != TK (max|diff| {d_bf:g})"
    Zb = np.asarray(npz[EMISSION_RUN], np.float64)
    assert Zb.shape[0] == N_STATES and float(np.abs(Zb[:, -1] - 1.0).max()) < 1e-5, \
        "the banked emission's homogeneous coordinate is not pinned to 1"
    c1j, prov = c1_provenance()
    med_npz = float(np.median(np.asarray(npz["cos_full_seed1"], np.float64)))
    assert abs(med_npz - prov["full_seed1_median_cos_json"]) < 1e-6, (
        "the npz's own full/seed1 cosines disagree with the reported run -- the emission array and "
        "the reported run are not the same object")
    prov["full_seed1_median_cos_from_npz"] = med_npz
    # the banked ingredient lists must BE the ones this rebuild produced
    sm = c1j["stage0_meta_m5"]
    assert len(sm) == len(meta)
    for a, b in zip(sm, meta):
        assert a["sid"] == b["sid"]
        assert list(a["top_m_pool_idx"]) == list(b["top_m_pool_idx"]), f"{a['sid']}: ingredients"
        assert max(abs(x - y) for x, y in zip(a["weights"], b["weights"])) < 1e-12
    log(f"[gate] m={M} targets bit-identical; emission {EMISSION_RUN} {Zb.shape} traced to C1 rev "
        f"{prov['c1_git_rev'][:8]} (median cos {med_npz:.7f}); stage0 ingredient lists match")

    # ---- R_s ---------------------------------------------------------------------------------------
    R, r_disc = reference_directions(br, meta, pool, pool2full)
    log(f"[reference] R_s {R.shape} built from {r_disc['n_item_ingredients']} item + "
        f"{r_disc['n_concept_ingredients']} concept ingredients; mean ||Ew|| "
        f"{r_disc['mean_item_tower_norm']:.4f} vs ||d_c|| {r_disc['mean_concept_tower_norm']:.4f}")

    # ---- the points ---------------------------------------------------------------------------------
    P = {"target": TKt[:N_STATES].astype(np.float64),
         "emission": Zb[:, :-1].copy()}
    P_all = np.concatenate([P["target"], P["emission"]], 0)
    P["random"], spec = spectrum_matched(P_all, P["target"], mu, N_STATES, RANDOM_SEED)
    spec["emission_norms_mean_removed"] = [float(x) for x in
                                           np.linalg.norm(P["emission"] - mu, axis=1)]
    spec["target_vs_emission_mean_removed_norm_ratio"] = float(np.mean(
        np.linalg.norm(P["emission"] - mu, axis=1) / np.linalg.norm(P["target"] - mu, axis=1)))
    spec["cos_target_emission_mean_removed_mean"] = float(np.mean(
        C3B.rowcos(P["target"] - mu, P["emission"] - mu)))
    for k in KINDS:
        assert P[k].shape == (N_STATES, Kpool.shape[1]) and np.isfinite(P[k]).all()
    log(f"[points] {len(KINDS)} kinds x {N_STATES} states; random rank {spec['rank']}, rescaled in "
        f"{spec['rescale_space']}; mean cos(target, emission) mean-removed "
        f"{spec['cos_target_emission_mean_removed_mean']:.5f}")

    # ---- the renderers -------------------------------------------------------------------------------
    C3B.ensure_lm()
    tok, lm = C3B.load_lm()
    torch.set_num_threads(int(args.threads))
    torch.manual_seed(SEED); np.random.seed(SEED)
    d_in = int(zp["X"].shape[1])
    ad = {}
    bl = {}
    for arm, fn in (("A", "c3b_true_best.pt"), ("Aprime", "c3b_scrambled_best.pt")):
        ad[arm], bl[arm] = C3B.load_adapter(os.path.join(C3B.OUT_DIR, fn), d_in)
        assert bl[arm]["split_hashes"] == man["split"]["hashes"], (
            f"the {arm} adapter was trained under a DIFFERENT split than the manifest's")
    log(f"[adapters] A ep{bl['A']['epoch']} val CE {bl['A']['val_ce']:.5f} | A' "
        f"ep{bl['Aprime']['epoch']} val CE {bl['Aprime']['val_ce']:.5f}; split hashes match")

    # ---- generation: 3 kinds x 2 channel one-hots x 2 arms x 42 states ---------------------------
    texts, ngen = {}, 0
    for kind in KINDS:
        for ch, chname in CHANNELS:
            Xin = C3B.input_vectors(P[kind], mu, scale, np.full(N_STATES, ch, np.int64))
            assert Xin.shape == (N_STATES, d_in)
            for arm in ARMS:
                out = []
                for st in range(0, N_STATES, args.gen_batch):
                    xb = torch.from_numpy(Xin[st:st + args.gen_batch])
                    with torch.no_grad():
                        out += C3B.greedy(lm, tok, ad[arm](xb))
                assert len(out) == N_STATES
                texts[(kind, chname, arm)] = out
                ngen += len(out)
            log(f"  [gen] {kind}/{chname}: A {len(set(texts[(kind, chname, 'A')]))} distinct / "
                f"A' {len(set(texts[(kind, chname, 'Aprime')]))} distinct "
                f"({(time.time() - t00) / 60:.1f}m)")

    # ---- the prior-only floor: zeroed latent + channel one-hot (the stage-2 construction) --------
    prior = {}
    for arm in ARMS:
        with torch.no_grad():
            pri = C3B.greedy(lm, tok, ad[arm](torch.from_numpy(
                C3B.zero_latent_input(d_in, [0, 1]))))
        prior[arm] = {"item_hot": pri[0], "concept_hot": pri[1]}
        ngen += 2
    log(f"[gen] prior-only  A: item={prior['A']['item_hot']!r} "
        f"concept={prior['A']['concept_hot']!r} | A': item={prior['Aprime']['item_hot']!r} "
        f"concept={prior['Aprime']['concept_hot']!r}")

    # ---- the catalogue comparator: the state's nearest bank question, GOLD phrase ----------------
    snap = np.asarray(npz[f"target_m{M}_snap"][:N_STATES], np.int64)
    assert (snap >= 0).all() and (snap < N_BANK).all()
    gold = [POC.phrase(br.names, ni, int(s)) for s in snap]
    gold_cos1 = np.asarray(npz[f"target_m{M}_cos1"][:N_STATES], np.float64)

    # ---- measure: every text through the bridge, cos to its state's R_s --------------------------
    def measure(txts):
        sel = br.select(list(txts))
        U = br.tower_vectors(sel)
        return C3B.rowcos(U, R), np.asarray([len(s) for s in sel], np.int64), U

    cos, nsel = {}, {}
    for key, tx in texts.items():
        cos[key], nsel[key], _ = measure(tx)
    cos[("catalogue", "gold", "-")], nsel[("catalogue", "gold", "-")], _ = measure(gold)
    for arm in ARMS:
        for _ch, chname in CHANNELS:
            k = ("prior", chname, arm)
            cos[k], nsel[k], _ = measure([prior[arm][chname]] * N_STATES)
    log(f"[encode] {ngen} generations + {N_STATES} catalogue phrases measured "
        f"({(time.time() - t00) / 60:.1f}m)")

    # ---- the readings ------------------------------------------------------------------------------
    def paired(a, b):
        """Paired difference over the 42 states, dropping only pairs where a side did not encode."""
        m = np.isfinite(a) & np.isfinite(b)
        out = dict(C3B.boot_mean((a - b)[m], seed=BOOT_SEED))
        out["n_pairs"] = int(m.sum())
        out["n_dropped_non_encodable"] = int((~m).sum())
        out["directional_pass"] = bool(out["mean"] is not None and out["mean"] > 0
                                       and out["ci_lo"] is not None and out["ci_lo"] > 0)
        return out

    def level(v):
        out = dict(C3B.boot_mean(v[np.isfinite(v)], seed=BOOT_SEED))
        out["n_non_encodable"] = int((~np.isfinite(v)).sum())
        return out

    r1, r2, levels = {}, {}, {}
    for kind in KINDS:
        for _ch, chname in CHANNELS:
            cell = f"{kind}/{chname}"
            r1[cell] = paired(cos[(kind, chname, "A")], cos[(kind, chname, "Aprime")])
            levels[cell] = {arm: level(cos[(kind, chname, arm)]) for arm in ARMS}
    for kind in REAL_KINDS:
        for _ch, chname in CHANNELS:
            cell = f"{kind}/{chname}"
            r2[cell] = paired(cos[(kind, chname, "A")], cos[("random", chname, "A")])

    r3 = {"level_gold_phrase_vs_R": level(cos[("catalogue", "gold", "-")]),
          "A_minus_catalogue": {f"{k}/{c}": paired(cos[(k, c, "A")], cos[("catalogue", "gold", "-")])
                                for k in REAL_KINDS for _x, c in CHANNELS},
          "bar": "NONE -- report-only (DS_C4_PILOT.md reading 3)",
          "nearest_bank_cos1_in_kphi_mean": float(gold_cos1.mean())}
    r4 = {f"{arm}/{chname}": level(cos[("prior", chname, arm)])
          for arm in ARMS for _x, chname in CHANNELS}

    # not registered, carried as diagnostics so no post-hoc reading can be dressed as a criterion
    diag = {"Aprime_real_minus_random": {
        f"{k}/{c}": paired(cos[(k, c, "Aprime")], cos[("random", c, "Aprime")])
        for k in REAL_KINDS for _x, c in CHANNELS},
        "A_minus_prior": {f"{k}/{c}": paired(cos[(k, c, "A")], cos[("prior", c, "A")])
                          for k in KINDS for _x, c in CHANNELS},
        "target_minus_emission_under_A": {
            c: paired(cos[("target", c, "A")], cos[("emission", c, "A")]) for _x, c in CHANNELS},
        "n_distinct_texts": {f"{k}/{c}/{a}": int(len(set(texts[(k, c, a)])))
                             for k in KINDS for _x, c in CHANNELS for a in ARMS},
        "mean_concepts_selected": {f"{k}/{c}/{a}": float(nsel[(k, c, a)].mean())
                                   for k in KINDS for _x, c in CHANNELS for a in ARMS}}

    # ---- the demo table ------------------------------------------------------------------------------
    demo, n_path_bad = [], []
    for i in range(N_STATES):
        s = reg[i]
        steps, ok = POC.path_in_words(s, br.names, ni, pool2full)
        if not ok:
            n_path_bad.append(s["sid"])
        row = {"i": i, "sid": s["sid"], "group": s["group"], "n_users": len(s["rows"]),
               "depth": len(s["prefix"]), "interview_path": steps,
               "catalogue_nearest_bank_question": {
                   "bank_idx": int(snap[i]),
                   "channel": "concept" if int(snap[i]) >= ni else "item",
                   "gold_phrase": gold[i],
                   "cos_to_R": None if not np.isfinite(cos[("catalogue", "gold", "-")][i])
                   else float(cos[("catalogue", "gold", "-")][i])},
               "renders": {}}
        for kind in KINDS:
            for _x, chname in CHANNELS:
                for arm in ARMS:
                    c = cos[(kind, chname, arm)][i]
                    row["renders"][f"{kind}/{chname}/{arm}"] = {
                        "text": texts[(kind, chname, arm)][i],
                        "cos_to_R": None if not np.isfinite(c) else float(c)}
        demo.append(row)

    # ---- output --------------------------------------------------------------------------------------
    out = {
        "analysis": "paperC_C4_pilot",
        "design_sheet": "docs/design/DS_C4_PILOT.md (PRE-REGISTERED 2026-08-25, BINDING)",
        "produced_by": "src/instrument/c4_pilot.py",
        "citable": True, "test_touched": False, "split": "validation",
        "answer_free": ("no answer is simulated to any rendered question, no NDCG is computed, and "
                        "no claim about the value of asking may be built from this file"),
        "n_states": N_STATES, "n_generations": int(ngen),
        "point_kinds": list(KINDS),
        "channel_hots": [c for _x, c in CHANNELS],
        "reference": {"definition": ("R_s = sum_j w_j * tower(q_j) over the state's banked m=5 "
                                     "ingredients; tower(.) = TowerBridge.target_tower (item -> Ew "
                                     "centred+popularity-stripped decoder row, concept -> d_c)"),
                      "same_R_for_target_emission_and_random": True,
                      "disclosure": r_disc},
        "measure": ("rendered text -> SBERT -> concepts -> ConceptFoldNet(cv3) fold displacement "
                    "from the cold fold z0, popularity-stripped -> cos with R_s"),
        "reading1_faithfulness_A_minus_Aprime": r1,
        "reading2_point_specificity_real_minus_random": r2,
        "reading3_catalogue_comparator_REPORT_ONLY": r3,
        "reading4_prior_floor": r4,
        "levels_cos_to_R_by_cell": levels,
        "diagnostics_not_registered": diag,
        "prior_only_texts": prior,
        "demo_table": demo,
        "adapters": {arm: {k: bl[arm][k] for k in ("epoch", "val_ce", "train_ce")} for arm in ARMS},
        "random_control": spec,
        "bootstrap": {"n_resamples": C3B.N_BOOT, "seed": BOOT_SEED,
                      "kind": "paired percentile over the 42 states, within cell, never pooled"},
        "reencode_recipe": {"source": "c3_poc_reencode via c3b_adapter.TowerBridge (IMPORTED)",
                            "sbert_model": br.RE.SBERT_MODEL, "pooling": br.RE.SBERT_MODE,
                            "topn": br.RE.SBERT_TOPN, "mincos": br.RE.SBERT_MINCOS,
                            "max_new_tokens": C3B.MAX_NEW_TOKENS},
        "asserts": {"pool2full_index_map_max_abs_err": map_err,
                    "bank_mean_vs_manifest_max_abs_err": d_mu,
                    "target_rebuild_max_abs_diff": d,
                    "biasfree_identity_max_abs_diff": d_bf,
                    "concept_dirs_reproduced_worst_1_minus_abscos": br.dc_reproduction_worst,
                    "c1_emission_provenance": prov,
                    "states_whose_sid_path_did_not_match_the_prefix": n_path_bad},
        "canonical_snap": {"val_m0_full": f0, "val_m0_tail": t0_},
        "checkpoint": {"path": os.path.relpath(args.ckpt, _ROOT), "sha256": ck_sha},
        "c3b_split_hashes": man["split"]["hashes"],
        "what_may_not_be_claimed": ("no NDCG, no 'asking it would help', no answer simulated, no "
                                    "human-plausibility grading, no pooling across the declared "
                                    "cells, no post-hoc primary; n = 42 nested, "
                                    "structurally-selected states (the C1 caveats travel)"),
        "git": GB,
        "seed_random_points": RANDOM_SEED,
        "minutes": round((time.time() - t00) / 60, 2)}
    with open(OUT_JSON + ".tmp", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, default=float)
    os.replace(OUT_JSON + ".tmp", OUT_JSON)

    arr = {"sid": np.asarray([s["sid"] for s in reg], object),
           "R": R, "snap": snap, "gold_phrase": np.asarray(gold, object),
           "cos_catalogue": cos[("catalogue", "gold", "-")]}
    for k in KINDS:
        arr[f"point_{k}"] = P[k].astype(np.float32)
    for key, tx in texts.items():
        tag = "_".join(key)
        arr[f"text_{tag}"] = np.asarray(tx, object)
        arr[f"cos_{tag}"] = cos[key]
        arr[f"nsel_{tag}"] = nsel[key]
    for arm in ARMS:
        for _x, chname in CHANNELS:
            arr[f"cos_prior_{chname}_{arm}"] = cos[("prior", chname, arm)]
            arr[f"text_prior_{chname}_{arm}"] = np.asarray([prior[arm][chname]], object)
    savez_atomic(OUT_NPZ, **arr)

    # ---- the console read ------------------------------------------------------------------------
    log("=" * 100)
    log("READING 1 -- FAITHFULNESS  A(point) - A'(point), per point-kind x channel-hot (directional)")
    for kind in KINDS:
        for _x, chname in CHANNELS:
            c = r1[f"{kind}/{chname}"]
            lv = levels[f"{kind}/{chname}"]
            log(f"  {kind:>9}/{chname:<12} d {c['mean']:+.5f} CI [{c['ci_lo']:+.5f}, "
                f"{c['ci_hi']:+.5f}] n={c['n_pairs']} -> "
                f"{'PASS' if c['directional_pass'] else 'no'}   "
                f"(A {lv['A']['mean']:+.5f}, A' {lv['Aprime']['mean']:+.5f})")
    log("READING 2 -- POINT-SPECIFICITY  A(real) - A(random), same channel (directional, uncertain)")
    for kind in REAL_KINDS:
        for _x, chname in CHANNELS:
            c = r2[f"{kind}/{chname}"]
            log(f"  {kind:>9}/{chname:<12} d {c['mean']:+.5f} CI [{c['ci_lo']:+.5f}, "
                f"{c['ci_hi']:+.5f}] n={c['n_pairs']} -> "
                f"{'PASS' if c['directional_pass'] else 'no'}")
    g = r3["level_gold_phrase_vs_R"]
    log(f"READING 3 -- CATALOGUE COMPARATOR (report-only, no bar): nearest bank question's gold "
        f"phrase vs R  {g['mean']:+.5f} CI [{g['ci_lo']:+.5f}, {g['ci_hi']:+.5f}]")
    for k, v in r3["A_minus_catalogue"].items():
        log(f"    A({k}) - catalogue {v['mean']:+.5f} CI [{v['ci_lo']:+.5f}, {v['ci_hi']:+.5f}]")
    log("READING 4 -- PRIOR FLOOR (prior-only text vs R; must sit at ~0)")
    for k, v in r4.items():
        log(f"  {k:<20} {v['mean']:+.5f} CI [{v['ci_lo']:+.5f}, {v['ci_hi']:+.5f}]")
    log("=" * 100)
    log(f"[done] {OUT_JSON} + {os.path.basename(OUT_NPZ)}  ({out['minutes']:.1f}m, {ngen} "
        f"generations)")
    if n_path_bad:
        log(f"[WARN] {len(n_path_bad)} state(s) whose sid path length != prefix length: {n_path_bad}")
    return out


def main():
    ap = argparse.ArgumentParser(description="Paper C / C4 pilot -- docs/design/DS_C4_PILOT.md")
    ap.add_argument("--ckpt", default=E1_CKPT)
    ap.add_argument("--snapshot", default=None)
    ap.add_argument("--concept_ckpt", default=None)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--gen_batch", type=int, default=42)
    ap.add_argument("--allow_dirty", action="store_true",
                    help="EMERGENCY ONLY: run with an uncommitted src/instrument (HARD RULE 10)")
    args = ap.parse_args()
    args.threads = min(int(args.threads), 4)
    from strategy_channel_suite import CERT_TOWER, CERT_CONC
    args.snapshot = args.snapshot or CERT_TOWER
    args.concept_ckpt = args.concept_ckpt or CERT_CONC
    log("#" * 100)
    log("# PAPER C / C4 PILOT -- render the policy's own emitted points and fold them back. "
        "ANSWER-FREE: no answer simulated, no NDCG, no test user built, loaded or touched.")
    log("#" * 100)
    return run(args)


if __name__ == "__main__":
    main()
