r"""PAPER C / C1 -- BLEND COHERENCE of the pilot's targets (author question, 2026-08-21).

Read-only analysis: no training, no new measurement, validation only. Rebuilds the SAME e_q bank the
pilot used (`c0_geometry.bank_vectors` on the C0-sha'd checkpoint) and recomputes the top-m ingredient
geometry, asserting the rebuilt targets are bit-identical to the pilot's before any statistic is
taken.

THE QUESTION. Stage 1's loss is `1 - cos`, which is norm-invariant, so it cannot see whether the
top-m ingredients of a target point the same way or cancel. If they cancel, `sum_j w_j e_{q_j}` is a
short, near-arbitrary vector and "the network fitted the target" means much less.

★ REPAIRED 2026-08-24 (hostile-review pass). Four changes, no change to any measured quantity:
  1. the thread caps are set BEFORE numpy/torch are imported (they were set after, so the caps did
     nothing and the run took whatever BLAS decided -- on a 4-physical-core box shared with other
     work that is not a cosmetic bug);
  2. the output is banked under `experiments/paperC/`, not a session scratchpad that disappears;
  3. every statistic `C1_PILOT.md` quotes is now COMPUTED HERE and banked -- the beat-the-floor
     counts per m and space, the mixed-vs-single-channel split with its Mann-Whitney test, the
     negative-pair counts and the low-coherence tail. They were previously derived by hand from
     `per_state` and existed only as prose;
  4. the per-m/per-space cells flush incrementally (HARD RULE 11).
The random stream, its consumption order and every formula are unchanged, so the banked numbers are
the doc's numbers.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)
if "--threads" not in sys.argv:                 # BEFORE numpy/torch: after them it is a no-op
    sys.argv += ["--threads", "3"]
_NT = sys.argv[sys.argv.index("--threads") + 1]
os.environ["OMP_NUM_THREADS"] = _NT
os.environ.setdefault("MKL_NUM_THREADS", _NT)
os.environ.setdefault("OPENBLAS_NUM_THREADS", _NT)

import json
import time
import argparse
import numpy as np
import torch

from policy_net import Policy, build_phi, load_traj, log, OUTDIR
from c0_geometry import PAPERC, E1_CKPT, N_BANK, sha256, git_block, index_map_assert, bank_vectors
import c1_pilot as C1

OUT = os.path.join(PAPERC, "c1_blend_coherence.json")
NDRAW = 200                      # random m-tuples per state, for the floor
SEED = 98765


def unit(X):
    return X / (np.linalg.norm(X, axis=-1, keepdims=True) + 1e-30)


def blend_stats(Ei, w):
    """Ei: (m, d) ingredient embeddings, w: (m,) weights summing to 1.

    `ratio` = ||sum_j w_j e_j|| / sum_j w_j ||e_j||: 1 = perfectly aligned ingredients, 0 = perfect
    cancellation. The pairwise block describes the same thing from the ingredient side."""
    num = np.linalg.norm((w[:, None] * Ei).sum(0))
    den = float((w * np.linalg.norm(Ei, axis=1)).sum())
    C = unit(Ei) @ unit(Ei).T
    iu = np.triu_indices(len(w), 1)
    pw = C[iu]
    W = np.outer(w, w)[iu]        # weight-aware pairwise mean
    return dict(ratio=float(num / (den + 1e-30)), pw_mean=float(pw.mean()),
                pw_min=float(pw.min()), pw_wmean=float((pw * W).sum() / W.sum()),
                pw_frac_neg=float((pw < 0).mean()), n_neg_pairs=int((pw < 0).sum()),
                n_pairs=int(len(pw)))


def bank_mean_pairwise(X):
    """Mean pairwise cosine over the WHOLE bank, in closed form (n^2 pairs are never materialised)."""
    U = unit(X.astype(np.float64))
    n = len(U)
    mu = U.mean(0)
    return float((n * float(mu @ mu) - 1.0) / (n - 1))


def q(x):
    x = np.asarray(x, np.float64)
    p = np.percentile(x, [0, 25, 50, 75, 100])
    return {"n": int(len(x)), "mean": float(x.mean()), "min": float(p[0]), "q1": float(p[1]),
            "median": float(p[2]), "q3": float(p[3]), "max": float(p[4])}


def summarise_rows(rows):
    """Everything C1_PILOT.md's coherence tables quote, for one (m, space) cell and one state set."""
    if not rows:
        return None
    o = {"n_states": len(rows)}
    for view in ("raw", "centred"):
        blk = {}
        for stat in ("ratio", "pw_mean", "pw_min", "pw_wmean"):
            blk[stat] = q([r[view][stat] for r in rows])
            blk["floor_" + stat] = q([r["rand_" + view][stat] for r in rows])
        # ★ the beat-the-floor counts: each state against ITS OWN random-m-tuple floor
        for stat in ("ratio", "pw_mean"):
            k = int(sum(r[view][stat] > r["rand_" + view][stat] for r in rows))
            blk[f"n_states_{stat}_beats_own_floor"] = k
            blk[f"frac_states_{stat}_beats_own_floor"] = k / len(rows)
        # ★ the negative-pair counts
        blk["n_states_with_a_negative_pair"] = int(sum(r[view]["pw_min"] < 0 for r in rows))
        blk["n_negative_pairs_total"] = int(sum(r[view]["n_neg_pairs"] for r in rows))
        blk["min_pairwise_over_states"] = float(min(r[view]["pw_min"] for r in rows))
        blk["lowest_ratio_states"] = [
            {"sid": r["sid"], "ratio": r[view]["ratio"], "pw_min": r[view]["pw_min"]}
            for r in sorted(rows, key=lambda r: r[view]["ratio"])[:5]]
        o[view] = blk
    return o


def channel_split(rows, m):
    """★ The mixed-vs-single-channel comparison C1_PILOT.md states with a p-value.

    'Single-channel' = the top-m ingredients are all items or all concepts; 'mixed' = anything else.
    Mann-Whitney U (two-sided, `scipy.stats.mannwhitneyu`) on the MEAN-REMOVED statistics, which are
    the only informative ones (the raw space is ~all common-mode component)."""
    from scipy.stats import mannwhitneyu, spearmanr
    nc = np.asarray([r["n_concept"] for r in rows])
    single = np.isin(nc, [0, m])
    out = {"n_single_channel": int(single.sum()), "n_mixed": int((~single).sum()),
           "concept_count_histogram": {str(k): int((nc == k).sum()) for k in range(m + 1)}}
    for view in ("raw", "centred"):
        blk = {}
        for stat in ("ratio", "pw_mean"):
            x = np.asarray([r[view][stat] for r in rows], np.float64)
            a, b = x[single], x[~single]
            if len(a) and len(b):
                u, p = mannwhitneyu(a, b, alternative="two-sided")
                blk[stat] = {"single_channel_median": float(np.median(a)),
                             "mixed_median": float(np.median(b)),
                             "single_channel_mean": float(a.mean()), "mixed_mean": float(b.mean()),
                             "mannwhitney_u": float(u), "mannwhitney_p_two_sided": float(p)}
            sp = spearmanr(nc, x)
            blk[stat + "_vs_n_concept"] = {"spearman": float(sp[0]), "p_spearman": float(sp[1])}
            blk[stat + "_by_n_concept"] = {
                str(k): {"n": int((nc == k).sum()),
                         "median": float(np.median(x[nc == k])) if (nc == k).any() else None}
                for k in range(m + 1)}
        out[view] = blk
    return out


def run(args):
    from strategy_channel_suite import build_greedy_env, CERT_TOWER, CERT_CONC
    from ndcg_tree import build_pool
    t00 = time.time()
    os.makedirs(PAPERC, exist_ok=True)
    GB = git_block(allow_dirty=args.allow_dirty, threads=args.threads)      # HARD RULE 10, first
    log(f"[hardrule10] src/instrument clean={GB['src_instrument_clean']} rev={GB['rev'][:8]} "
        f"threads={GB['threads']}")

    blob = torch.load(E1_CKPT, map_location="cpu")
    sha = sha256(E1_CKPT)
    assert sha == C1.C0_CKPT_SHA, f"this is not the C0/C1-measured checkpoint (sha {sha})"
    E = build_greedy_env(args.snapshot, "validation", args.concept_ckpt, "paperC/coherence")
    pool, _it, _cc = build_pool(E, 500, 500)
    pool = [(int(a), int(b)) for a, b in pool]
    assert pool == [(int(a), int(b)) for a, b in blob["pool"]]
    pool_index = {q_: i for i, q_ in enumerate(pool)}
    full_pool = [(0, i) for i in range(E["ctx"].ni)] + [(1, c) for c in range(E["C"])]
    assert len(full_pool) == N_BANK
    pool2full = np.asarray([(i if ch == 0 else E["ctx"].ni + i) for ch, i in pool], np.int64)
    norm = (blob["norm_mu"], blob["norm_sd"])
    phi_t, _ = build_phi(E, pool, norm=norm)
    phi_full, _ = build_phi(E, full_pool, norm=norm)
    pol0 = Policy(E["ctx"].d, phi_full.shape[1], len(full_pool), arch=blob["arch"],
                  state=blob["state"], use_pos=blob["use_pos"])
    pol0.load_state_dict(blob["model"]); pol0.eval()
    Ebank, Kbank = bank_vectors(pol0, phi_full)
    Epool, Kpool = bank_vectors(pol0, phi_t)
    map_err = index_map_assert(Ebank, pool2full, Epool)
    log(f"[bank] Ebank {Ebank.shape} Epool {Epool.shape}; pool2full exact (max|diff| {map_err:g})")

    Dva = load_traj(train=False)
    reg, extra = C1.load_the_42(pool_index, Dva)
    states = reg + extra
    log(f"[states] {len(reg)} registered + {len(extra)} secondary")

    # --- the pilot's own targets, reproduced bit-for-bit, as a gate --------------------------------
    npz = np.load(os.path.join(PAPERC, "c1_pilot_per_state.npz"))
    gate = {}
    for m in (3, 5, 10):
        Tt, _TKt, _meta = C1.build_targets(states, Epool, Kpool, pool_index, m)
        d = float(np.abs(Tt - npz[f"target_m{m}"]).max())
        log(f"[gate] m={m} target reproduction max|diff| {d:.3g}")
        assert d == 0.0, "the rebuilt targets are not the pilot's targets"
        gate[f"m{m}_target_max_abs_diff_vs_pilot"] = d

    CH = np.asarray([c for c, _ in pool], np.int64)          # 0 = item, 1 = concept

    spaces = {}
    for name, Bank, Pool in (("homogeneous", Ebank, Epool), ("bias_free", Kbank, Kpool)):
        mu = Bank.astype(np.float64).mean(0)
        spaces[name] = dict(bank=Bank, pool=Pool, mu=mu,
                            bank_pw_raw=bank_mean_pairwise(Bank),
                            bank_pw_centred=bank_mean_pairwise(Bank - mu),
                            mu_norm_ratio=float(np.linalg.norm(mu) /
                                                np.linalg.norm(Bank, axis=1).mean()))
        log(f"[bank anisotropy/{name}] mean pairwise cos raw {spaces[name]['bank_pw_raw']:.4f} "
            f"centred {spaces[name]['bank_pw_centred']:.4f} "
            f"| ||mu||/mean||e|| {spaces[name]['mu_norm_ratio']:.4f}")

    out = {"analysis": "paperC_C1_blend_coherence",
           "design_sheet": "docs/design/DS_C1_PILOT.md",
           "produced_by": "src/instrument/c1_blend_coherence.py",
           "citable": True, "test_touched": False, "split": "validation",
           "checkpoint": {"path": os.path.relpath(E1_CKPT, _ROOT), "sha256": sha},
           "git": GB,
           "asserts": dict(gate, pool2full_index_map_max_abs_err=map_err),
           "n_random_draws": NDRAW, "seed": SEED,
           "bank": {k: {kk: v[kk] for kk in ("bank_pw_raw", "bank_pw_centred", "mu_norm_ratio")}
                    for k, v in spaces.items()},
           "summary": {}, "channel_split": {}, "correlations": {}, "per_state": {}}

    def flush():
        out["minutes"] = round((time.time() - t00) / 60, 2)
        with open(OUT + ".tmp", "w") as fh:
            json.dump(out, fh, indent=1, default=float)
        os.replace(OUT + ".tmp", OUT)

    flush()

    # --- the per-state geometry. The rng stream and its CONSUMPTION ORDER are unchanged from the
    # --- 2026-08-21 run, so these are the numbers C1_PILOT.md quotes.
    rng = np.random.default_rng(SEED)
    per_state = {}
    for m in (3, 5, 10):
        _T, _TK, meta = C1.build_targets(states, Epool, Kpool, pool_index, m)
        for sname in ("homogeneous", "bias_free"):
            P = spaces[sname]["pool"]
            mu = spaces[sname]["mu"].astype(np.float32)
            Pc = P - mu
            rows = []
            for i, s in enumerate(states):
                qi = np.asarray(meta[i]["top_m_pool_idx"], np.int64)
                w = np.asarray(meta[i]["weights"], np.float64)
                asked = {pool_index[a] for a in s["prefix"]}
                avail = np.asarray([j for j in range(len(pool)) if j not in asked], np.int64)
                raw = blend_stats(P[qi].astype(np.float64), w)
                cen = blend_stats(Pc[qi].astype(np.float64), w)
                rr, rc = [], []
                for _ in range(NDRAW):
                    r = rng.choice(avail, size=m, replace=False)
                    rr.append(blend_stats(P[r].astype(np.float64), w))
                    rc.append(blend_stats(Pc[r].astype(np.float64), w))
                agg = lambda L, k: float(np.mean([x[k] for x in L]))
                rows.append(dict(sid=s["sid"], registered=bool(s.get("registered", False)),
                                 n_concept=int((CH[qi] == 1).sum()),
                                 n_item=int((CH[qi] == 0).sum()),
                                 raw=raw, centred=cen,
                                 rand_raw={k: agg(rr, k) for k in raw},
                                 rand_centred={k: agg(rc, k) for k in cen}))
            cell = f"m{m}/{sname}"
            per_state[cell] = rows
            out["per_state"][cell] = rows
            R = [r for r in rows if r["registered"]]
            S = [r for r in rows if not r["registered"]]
            out["summary"][cell] = {"registered42": summarise_rows(R),
                                    "secondary_unmapped7": summarise_rows(S)}
            out["channel_split"][cell] = channel_split(R, m)
            flush()                                        # HARD RULE 11: one cell per flush
            sm = out["summary"][cell]["registered42"]
            log(f"[m={m} {sname}] n={len(R)} ratio med {sm['raw']['ratio']['median']:.4f} "
                f"| rand {sm['raw']['floor_ratio']['median']:.4f} "
                f"| centred ratio med {sm['centred']['ratio']['median']:.4f} "
                f"(floor {sm['centred']['floor_ratio']['median']:.4f}, beats own floor at "
                f"{sm['centred']['n_states_ratio_beats_own_floor']}/{len(R)}) "
                f"| centred pw med {sm['centred']['pw_mean']['median']:.4f} "
                f"(beats floor {sm['centred']['n_states_pw_mean_beats_own_floor']}/{len(R)}) "
                f"| states with a negative pair {sm['centred']['n_states_with_a_negative_pair']}")

    # --- correlations ------------------------------------------------------------------------------
    from scipy.stats import spearmanr, pearsonr
    corr = {}
    for m in (3, 5, 10):
        for sname in ("homogeneous", "bias_free"):
            rows = [r for r in per_state[f"m{m}/{sname}"] if r["registered"]]
            assert len(rows) == 42
            x = np.asarray([r["raw"]["ratio"] for r in rows])
            xc = np.asarray([r["centred"]["ratio"] for r in rows])
            c = {}
            tgt = np.asarray(npz[f"target_m{m}_cos1"])[:42]
            c["target_cos1"] = dict(pearson=float(pearsonr(x, tgt)[0]),
                                    spearman=float(spearmanr(x, tgt)[0]),
                                    p_spearman=float(spearmanr(x, tgt)[1]))
            c["target_cos1_centred_ratio"] = dict(spearman=float(spearmanr(xc, tgt)[0]),
                                                  p_spearman=float(spearmanr(xc, tgt)[1]))
            c["target_margin"] = dict(
                spearman=float(spearmanr(x, np.asarray(npz[f"target_m{m}_m"])[:42])[0]))
            if m == 5:
                for run_ in ("head_only_seed1", "full_seed1", "head_only_seed2", "full_seed2"):
                    f = np.asarray(npz[f"cos_{run_}"])
                    c[f"fit_{run_}"] = dict(pearson=float(pearsonr(x, f)[0]),
                                            spearman=float(spearmanr(x, f)[0]),
                                            p_spearman=float(spearmanr(x, f)[1]))
                    c[f"fit_{run_}_centred_ratio"] = dict(
                        pearson=float(pearsonr(xc, f)[0]),
                        spearman=float(spearmanr(xc, f)[0]),
                        p_spearman=float(spearmanr(xc, f)[1]))
            nc = np.asarray([r["n_concept"] for r in rows])
            c["n_concept_vs_ratio"] = dict(spearman=float(spearmanr(nc, x)[0]),
                                           p_spearman=float(spearmanr(nc, x)[1]))
            corr[f"m{m}/{sname}"] = c
    out["correlations"] = corr
    flush()
    log(f"[done] wrote {OUT} in {out['minutes']:.1f} min")
    return out


def main():
    ap = argparse.ArgumentParser(description="Paper C / C1 blend coherence (read-only)")
    ap.add_argument("--snapshot", default=None)
    ap.add_argument("--concept_ckpt", default=None)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--allow_dirty", action="store_true",
                    help="EMERGENCY ONLY: run with an uncommitted src/instrument (HARD RULE 10)")
    args = ap.parse_args()
    from strategy_channel_suite import CERT_TOWER, CERT_CONC
    args.snapshot = args.snapshot or CERT_TOWER
    args.concept_ckpt = args.concept_ckpt or CERT_CONC
    log("#" * 100)
    log("# PAPER C / C1 BLEND COHERENCE -- read-only over banked arrays. VALIDATION ONLY; no "
        "training, no LLM call, no test user built, loaded or touched.")
    log("#" * 100)
    return run(args)


if __name__ == "__main__":
    main()
