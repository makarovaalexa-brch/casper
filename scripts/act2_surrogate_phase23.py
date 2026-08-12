r"""act2_surrogate_phase23.py -- ACT 2 / C1 / TODO 1.11, PHASES 2 AND 3.

Design sheet: `docs/design/DS_ACT2_SURROGATE_RANK_CORR.md` sections 4-8. Phase 1
(`act2_surrogate_rank_corr.py --score`) measures the REALISED NDCG@10 gain of every candidate at
every state and is the only expensive part. This file does the two cheap halves:

  PHASE 2  `--surrogates`  each row's own acquisition score at each state, via the `score_vector`
                           hook, with gate **G-F** asserted per row per state: the argmax of the
                           emitted vector must equal the action `select()` returns. The probe walks
                           the BANKED interview state; no interview is re-run and no answer is
                           re-simulated.
  PHASE 3  `--analyze`     the statistics, the three controls and the identity (all in
                           `act2_surrogate_stats.py`), then the results document.

WHY THE TWO ARE ONE FILE. They share the state reconstruction, the pool mapping and the on-disk
layout, and splitting them would mean a third copy of the (state -> users) map. Phase 1 stays in its
own file because it is the only thing that must survive a 12-hour unattended run.

THE ONE THING THAT COSTS ANYTHING IN PHASE 2 is constructing the selectors: the frozen iALS basis and
the Partial-VAE have to be loaded, and `load_arm_n` reads the split. That is minutes, not hours, and
it is why phase 2 is called cheap -- but it is not free, so it does not run while another job owns
the box either.

  python scripts/act2_surrogate_phase23.py --surrogates --budgets 1,2,0,4,8
  python scripts/act2_surrogate_phase23.py --analyze
  python scripts/act2_surrogate_phase23.py --selftest       # synthetic end-to-end, no box needed
"""
from __future__ import annotations

import os
import sys
import json
import time
import shutil
import argparse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "baselines"))
sys.path.insert(0, str(ROOT / "src" / "instrument"))
sys.path.insert(0, str(ROOT / "scripts" / "reddit"))
sys.path.insert(0, str(ROOT / "scripts"))

import act2_surrogate_stats as S                                          # noqa: E402

CACHE = ROOT / ".cache" / "act2_surrogate"
SURR = CACHE / "surrogates"
OUTJSON = ROOT / "experiments" / "baselines" / "interview" / "act2_surrogate_rank_corr.json"
DOC = ROOT / "docs" / "results" / "ACT2_SURROGATE_RANK_CORRELATION.md"
BUDGET_ORDER = (1, 2, 0, 4, 8)

# The 20 rows of design-sheet section 4. `vf` = expected VALUE-FREE at these states, which is a
# PREDICTION the run checks rather than a label it applies: `value_free_measured` in the output is
# computed from the emitted vectors, and a disagreement is a finding, not a formatting problem.
ROWS = [
    ("glm_ucb", "items", False), ("bald", "items", True), ("aopt", "items", True),
    ("thompson", "items", False), ("eddi_ig", "items", False), ("eddi_ig3_signed", "items", False),
    ("igcn", "items", True), ("dre", "items", True),
    ("popularity-static", "items", True), ("helf-static", "items", True),
    ("entropy0-static", "items", True), ("pure_entropy-static", "items", True),
    ("conlinucb_mcr", "both", False), ("conlinucb_bs", "both", False),
    ("conlinucb_bs_static", "concepts", True), ("clime", "both", False), ("conucb", "both", False),
    ("dpe_concept", "concepts", False),
]
# Rows with no per-candidate score at a user state. Printed in the table WITH the reason (design
# sheet section 4b) rather than omitted.
ABSENT = {"golbandi_tree": ("its split criterion is a within-node squared error over TRAIN users "
                            "evaluated during tree growth; `release()` frees those matrices and the "
                            "object at interview time does pure ternary routing, so no per-candidate "
                            "score exists at a test user's state")}


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ===================================================================== shared: the cell on disk
def load_cell(k, cdir=None):
    """Phase 1's banked cell for one budget. Everything phases 2 and 3 need, and nothing they don't:
    the realised gains, the state map, the pool, and the contract they were produced under."""
    d = Path(cdir) if cdir else (CACHE / f"k{k}")
    ctr = json.load(open(d / "contract.json"))
    pool = np.load(d / "pool.npy")
    sid = np.load(d / "state_id.npy")
    done = np.load(d / "done.npy")
    gf = np.load(d / "gain_full.npy", mmap_mode="r")
    gt = np.load(d / "gain_tail.npy", mmap_mode="r")
    assert bool(done.all()), (f"k={k}: {int((~done).sum())} of {len(done)} candidates are NOT "
                              f"scored; this cell is incomplete and must not be analysed")
    ns = int(ctr["n_states"])
    assert sid.max() + 1 == ns, f"k={k}: state map has {sid.max() + 1} states, contract says {ns}"
    return {"k": k, "contract": ctr, "pool": pool, "sid": sid, "ns": ns,
            "gain_full": gf, "gain_tail": gt, "dir": d}


def pool_channels(pool):
    """(n_pool,) 0 = item, 1 = concept -- the channel split every statistic is also reported over."""
    return np.asarray(pool)[:, 0].astype(np.int64)


def state_representatives(sid, ns):
    """One user id per state -- the LOWEST, deterministically, so a re-run probes the same users.

    Sound because a state is by construction the full interview history: every user in a state has
    the same asked set and the same answers, so every selector's score vector is identical across
    them. The randomised rows (thompson, dpe*) are the exception and are handled by seeding on the
    representative's own id, which is what makes their row reproducible rather than resample noise."""
    rep = np.full(ns, -1, np.int64)
    for s in range(ns):
        u = np.flatnonzero(sid == s)
        if len(u):
            rep[s] = int(u[0])
    assert (rep >= 0).all(), "a state has no users"
    return rep


# ===================================================================== PHASE 2
def phase_surrogates(args):
    """Emit each row's own acquisition vector at each banked state. GATE G-F fires per row per state.

    No interview is re-run: the asked/answered history comes from the banked `fixedorder:entropy0`
    checkpoints, exactly as phase 1's states do, through the same `read_interview`/`states_of` pair.
    """
    from adaptive_harness import build_selector, logln
    from arm_n import load_arm_n
    import interview_strategies as ST
    from act2_pv_quartiles import read_interview
    from act2_surrogate_rank_corr import build_states, PV_SPEC, CKDIR

    budgets = [int(x) for x in args.budgets.split(",")] if args.budgets else list(BUDGET_ORDER)
    D = load_arm_n(log=logln)
    ni = int(D["train"].shape[1])
    glob, bank = ST.build_orders(D["g_train"], D["train"], ni, log=logln)
    cache = {}
    manifest = {}

    for k in budgets:
        cell = load_cell(k)
        pool, sid, ns = cell["pool"], cell["sid"], cell["ns"]
        rep = state_representatives(sid, ns)
        prefix, sid2, ns2 = build_states(k)
        assert ns2 == ns and np.array_equal(sid2, sid), \
            (f"k={k}: the state map rebuilt now does not match the one phase 1 banked -- the banked "
             f"interviews have moved, and nothing may be analysed against a different partition")

        # the banked interview history, per user, straight off the harness's own checkpoint
        if k == 0:
            asked = {int(u): [] for u in range(len(sid))}
            answered = {int(u): [] for u in range(len(sid))}
        else:
            a_all, w_all, toks = read_interview(PV_SPEC, k, CKDIR)
            assert all(len(t) == 0 for t in toks), \
                (f"k={k}: the reference interview spends a turn on a CONCEPT. The concept-channel "
                 f"rows read their state from ctx['conc_state'], which this probe supplies as empty; "
                 f"a non-empty concept history would silently be dropped.")
            asked = {int(u): [int(q) for q in a_all[u]] for u in range(len(a_all))}
            answered = {int(u): [(int(i), int(lv)) for i, lv in w_all[u]] for u in range(len(w_all))}

        # The concept channel: EMPTY at these states by construction (the reference interview spends
        # zero concept turns -- asserted above, not assumed), but the concept-asking rows still need
        # the dict to exist and need `n_concepts` to size their action space. `n_concepts` is taken
        # from the concept stack the selector itself was built against, via `sel.C`, after
        # construction; the ctx value below is only the harness-level bound and must agree with it.
        conc_state = {"asked": {int(u): [] for u in range(len(sid))},
                      "answered": {int(u): {} for u in range(len(sid))}}
        n_conc_ctx = int(cache.get("n_concepts", 0))

        outdir = SURR / f"k{k}"
        outdir.mkdir(parents=True, exist_ok=True)
        manifest[k] = {}
        users = rep.astype(np.int64)

        for spec, channel, expect_vf in ROWS:
            if args.rows and spec not in args.rows.split(","):
                continue
            t0 = time.time()
            try:
                sel = build_selector(spec, D, glob, bank, max(k, 1), args, cache, log=logln)
            except Exception as exc:                      # noqa: BLE001 -- reported, never silent
                manifest[k][spec] = {"status": "CONSTRUCTION_FAILED", "error": repr(exc)}
                log(f"[k={k}] {spec:22s} CONSTRUCTION FAILED: {exc!r}")
                json.dump(manifest, open(SURR / "manifest.json", "w"), indent=1)
                continue
            n_conc = int(getattr(sel, "C", 0) or 0)
            if n_conc:
                cache["n_concepts"] = n_conc
            base_ctx = {"n_users": len(sid), "n_items": ni, "q_max": max(k, 1), "log": logln,
                        "D": D, "glob": glob, "bank": bank, "conc_state": conc_state,
                        "n_concepts": n_conc or n_conc_ctx}
            sel.reset(base_ctx)
            if channel in ("both", "concepts"):
                assert n_conc > 0, (f"{spec} asks concepts but exposes no concept count; its action "
                                    f"space cannot be sized and G-F would compare the wrong axes")

            V = sel.score_vector(k, users, asked, answered)
            if V is None:
                manifest[k][spec] = {"status": "NO_SCORE_AT_A_STATE"}
                log(f"[k={k}] {spec:22s} emits no per-candidate score at a user state")
                continue
            act = np.asarray(sel.select(k, users, asked, answered), dtype=np.int64)

            # ---------------------------------------------------------------- GATE G-F
            bad = []
            for j in range(len(users)):
                row = V[j]
                if not np.isfinite(row).any():            # declared-unscoreable state (IGCN)
                    continue
                if int(np.nanargmax(row)) != int(act[j]):
                    bad.append((int(users[j]), int(np.nanargmax(row)), int(act[j])))
            assert not bad, (f"G-F FAILED for {spec} at k={k}: {len(bad)} states where the emitted "
                             f"vector's argmax is not the action select() returns; first 3 {bad[:3]}")

            # restrict to the pool, mapping (channel, id) -> action index
            act_idx = np.asarray([(int(i) if int(c) == 0 else ni + int(i)) for c, i in pool],
                                 dtype=np.int64)
            keep = act_idx < V.shape[1]
            P = np.full((ns, len(pool)), np.nan, np.float32)
            P[:, keep] = V[:, act_idx[keep]].astype(np.float32)
            P[~np.isfinite(P)] = np.nan                   # NEG -> "this row cannot score this"

            vf = bool(np.all(np.isclose(P, P[0][None, :], equal_nan=True))) if ns > 1 else None
            np.savez_compressed(outdir / f"{spec}.npz", pool_scores=P, rep=users,
                                action=act.astype(np.int64))
            manifest[k][spec] = {"status": "OK", "channel": channel, "n_states": int(ns),
                                 "value_free_expected": expect_vf, "value_free_measured": vf,
                                 "gf_pass": True, "seconds": round(time.time() - t0, 1),
                                 "scoreable_states": int(np.isfinite(P).any(axis=1).sum())}
            if vf is not None and vf != expect_vf:
                manifest[k][spec]["★ DISAGREEMENT"] = (
                    f"section 4 predicts value_free={expect_vf}; the emitted vectors say {vf}")
                log(f"[k={k}] {spec:22s} ★ VALUE-FREE PREDICTION DISAGREES: expected {expect_vf}, "
                    f"measured {vf}")
            cov = getattr(sel, "score_vector_coverage", None)
            if cov is not None:
                manifest[k][spec]["coverage"] = cov()
            log(f"[k={k}] {spec:22s} OK  {ns} states, value-free={vf}, G-F PASS, "
                f"{time.time() - t0:.1f}s")
            json.dump(manifest, open(SURR / "manifest.json", "w"), indent=1)
    json.dump(manifest, open(SURR / "manifest.json", "w"), indent=1)
    log(f"[phase 2] manifest -> {SURR / 'manifest.json'}")
    return manifest


# ===================================================================== PHASE 3
def analyze_budget(k, axis="full", cdir=None, sdir=None, n_boot=S.N_BOOT, seed=S.BOOT_SEED):
    """All rows at one budget, plus the three controls and the identity."""
    cell = load_cell(k, cdir)
    sid, ns, pool = cell["sid"], cell["ns"], cell["pool"]
    ch = pool_channels(pool)
    gain_qu = np.asarray(cell["gain_full" if axis == "full" else "gain_tail"], np.float64)
    G = S.state_gain_matrix(gain_qu, sid, ns)
    w = np.bincount(sid, minlength=ns).astype(np.float64)

    out = {"budget": k, "axis": axis, "n_states": ns, "n_users": int(len(sid)),
           "n_pool": int(len(pool)), "contract": cell["contract"], "rows": {}, "controls": {}}

    # ---- CEILING (a property of the data, not of any row)
    rho_c, wc = S.split_half_ceiling(gain_qu, sid, ns, np.arange(len(sid)))
    ceil = S.weighted_mean(rho_c, wc)
    ceil.update(S.boot_states(rho_c, wc, n_boot, seed))
    ceil["defined_user_mass"] = float(wc[np.isfinite(rho_c)].sum() / max(wc.sum(), 1))
    out["controls"]["ceiling_split_half"] = ceil

    Gshuf = S.shuffle_gain(G, seed)
    Gswap = S.swap_gain(G, seed)
    out["controls"]["state_swap_applicable"] = Gswap is not None

    sdir = Path(sdir) if sdir else (SURR / f"k{k}")
    for spec, channel, expect_vf in ROWS:
        f = sdir / f"{spec}.npz"
        if not f.exists():
            out["rows"][spec] = {"status": "NOT_MEASURED"}
            continue
        z = np.load(f)
        A = np.asarray(z["pool_scores"], np.float64)
        r = {"status": "OK", "channel": channel,
             "primary": S.evaluate_row(A, G, w, ch, boot=True, n_boot=n_boot, seed=seed),
             "floor": S.evaluate_row(A, Gshuf, w, ch, boot=False)["combined"]["spearman"],
             "identity": S.self_difference_identity(A, w)}
        if Gswap is not None:
            r["state_swap"] = S.evaluate_row(A, Gswap, w, ch, boot=False)["combined"]["spearman"]
        # per-user regret, for the user-side bootstrap the design sheet asks for
        per_state_regret = r["primary"]["combined"]["per_state"]["regret"]
        r["regret_user_bootstrap"] = S.boot_users(np.asarray(per_state_regret)[sid], n_boot, seed)
        for scope in r["primary"]:
            r["primary"][scope].pop("per_state", None)
        out["rows"][spec] = r
    for spec, why in ABSENT.items():
        out["rows"][spec] = {"status": "NO_PER_CANDIDATE_SCORE_AT_A_STATE", "reason": why}

    # ---- GATE G-G: the floor's interval must cover 0 for every row
    bad = [s for s, r in out["rows"].items()
           if r.get("status") == "OK" and np.isfinite(r["floor"].get("mean", np.nan))
           and abs(r["floor"]["mean"]) > 0.05]
    out["controls"]["G_G_shuffle_floor_pass"] = not bad
    out["controls"]["G_G_offenders"] = bad
    return out


def phase_analyze(args):
    budgets = [int(x) for x in args.budgets.split(",")] if args.budgets else list(BUDGET_ORDER)
    res = {"analysis": "act2_surrogate_rank_correlation", "citable": True,
           "design_sheet": "docs/design/DS_ACT2_SURROGATE_RANK_CORR.md",
           "inference": (f"paired percentile bootstrap over states, {S.N_BOOT} resamples, "
                         f"seed {S.BOOT_SEED}; regret additionally bootstrapped over users"),
           "pool_restriction": ("1,000 candidates (500 popularity items + 500 answerable concepts); "
                                "author sign-off 2026-08-12; the 19,390-candidate alternative "
                                "(~230 h) was costed and declined; phase 1b makes REGRET pool-free"),
           "budgets": {}}
    for k in budgets:
        for axis in ("full", "tail"):
            try:
                res["budgets"][f"k{k}|{axis}"] = analyze_budget(k, axis, n_boot=args.boot,
                                                               seed=args.seed)
                log(f"[analyze] k={k} {axis}: done")
            except (FileNotFoundError, AssertionError) as exc:
                res["budgets"][f"k{k}|{axis}"] = {"status": "UNAVAILABLE", "why": repr(exc)}
                log(f"[analyze] k={k} {axis}: UNAVAILABLE -- {exc}")
    OUTJSON.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUTJSON, "w"), indent=1, default=float)
    log(f"[analyze] -> {OUTJSON}")
    return res


# ===================================================================== selftest
def _selftest():
    """END-TO-END on synthetic files, so the whole phase-3 path is exercised with the box held.

    Fabricates a phase-1 cell (memmaps, state map, pool, contract) and two phase-2 rows -- one
    ALIGNED with the realised gain and one CONSTANT -- then runs `analyze_budget` and checks that it
    recovers the properties it must: the aligned row near 1, the constant row UNDEFINED, the floor
    near 0, the identity exact, and the ceiling reported with its coverage."""
    log("SELFTEST: synthetic phase-1 cell + phase-2 rows -> analyze_budget")
    tmp = CACHE / "_selftest23"
    if tmp.exists():
        shutil.rmtree(tmp)
    cdir, sdir = tmp / "k1", tmp / "surr"
    cdir.mkdir(parents=True); sdir.mkdir(parents=True)
    rng = np.random.default_rng(3)
    nq, nu, ns = 60, 400, 9
    sid = rng.integers(0, ns, nu).astype(np.int64)
    sid[:ns] = np.arange(ns)                                  # every state non-empty
    gain = rng.normal(size=(nq, nu)).astype(np.float32)
    np.save(cdir / "gain_full.npy", gain)
    np.save(cdir / "gain_tail.npy", gain * 0.5)
    np.save(cdir / "done.npy", np.ones(nq, bool))
    np.save(cdir / "state_id.npy", sid)
    np.save(cdir / "pool.npy", np.stack([np.r_[np.zeros(nq // 2), np.ones(nq // 2)],
                                         np.arange(nq)], 1).astype(np.int64))
    json.dump({"n_states": ns, "budget": 1, "n_pool": nq}, open(cdir / "contract.json", "w"))

    G = S.state_gain_matrix(gain.astype(np.float64), sid, ns)
    np.savez_compressed(sdir / "glm_ucb.npz", pool_scores=G.astype(np.float32),
                        rep=np.arange(ns), action=np.zeros(ns, np.int64))
    np.savez_compressed(sdir / "bald.npz", pool_scores=np.zeros((ns, nq), np.float32),
                        rep=np.arange(ns), action=np.zeros(ns, np.int64))

    out = analyze_budget(1, "full", cdir=cdir, sdir=sdir, n_boot=200)
    a = out["rows"]["glm_ucb"]["primary"]["combined"]
    assert abs(a["spearman"]["mean"] - 1.0) < 1e-9, a["spearman"]
    assert abs(a["regret"]["mean"]) < 1e-12
    assert abs(a["top1"]["mean"] - 1.0) < 1e-12
    log(f"  aligned row: rho={a['spearman']['mean']:.6f} top1={a['top1']['mean']:.3f} "
        f"regret={a['regret']['mean']:.2e}                PASS")
    b = out["rows"]["bald"]["primary"]["combined"]["spearman"]
    assert not np.isfinite(b["mean"]) and b["n_defined"] == 0, b
    log(f"  constant row: UNDEFINED on all {b['n_states']} states, not 0.0                 PASS")
    fl = out["rows"]["glm_ucb"]["floor"]["mean"]
    assert abs(fl) < 0.35, fl
    log(f"  shuffle floor for the aligned row: {fl:+.4f} (near 0, not near 1)          PASS")
    assert out["rows"]["glm_ucb"]["identity"]["pass"], out["rows"]["glm_ucb"]["identity"]
    log("  self-difference identity exact                                          PASS")
    assert out["rows"]["golbandi_tree"]["status"] == "NO_PER_CANDIDATE_SCORE_AT_A_STATE"
    log("  golbandi_tree present in the table WITH its reason, not omitted          PASS")
    c = out["controls"]["ceiling_split_half"]
    assert "defined_user_mass" in c
    log(f"  ceiling reported with coverage: user mass {c['defined_user_mass']:.3f}           PASS")
    assert out["controls"]["state_swap_applicable"]
    assert "state_swap" in out["rows"]["glm_ucb"]
    log("  state swap applied against a derangement                                PASS")
    shutil.rmtree(tmp, ignore_errors=True)
    log("SELFTEST PASSED (phases 2/3 analysis path)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--surrogates", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--budgets", default="")
    ap.add_argument("--rows", default="")
    ap.add_argument("--boot", type=int, default=S.N_BOOT)
    ap.add_argument("--seed", type=int, default=S.BOOT_SEED)
    # consumed by build_selector for the arms that take them; defaults match the battery cells
    ap.add_argument("--conv_max_states", type=int, default=0)
    ap.add_argument("--concept_neg_cap", type=int, default=0)
    ap.add_argument("--keyterm_repr", default="")
    ap.add_argument("--snapshot", default="")
    ap.add_argument("--ucb_alpha", type=float, default=1.0)
    ap.add_argument("--ts_seed", type=int, default=98765)
    ap.add_argument("--eddi_cand_topm", type=int, default=0)
    a = ap.parse_args()
    if a.selftest:
        return _selftest()
    if a.surrogates:
        return phase_surrogates(a)
    if a.analyze:
        return phase_analyze(a)
    ap.error("choose one of --selftest / --surrogates / --analyze")


if __name__ == "__main__":
    main()
