r"""Blend coherence for the C1 pilot targets (author question, 2026-08-21). Read-only analysis:
no training, no new measurement, validation only. Rebuilds the SAME e_q bank the pilot used
(c0_geometry.bank_vectors on the C0-sha'd checkpoint) and recomputes the top-m ingredient geometry.
"""
import os, sys, json
_HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)))
SRC = r"C:\dev\phd\casper\src\instrument"
sys.argv += ["--threads", "3"]
sys.path.insert(0, SRC)
import numpy as np
import torch

from policy_net import Policy, build_phi, load_traj, log, TOTAL_L, OUTDIR
from c0_geometry import PAPERC, E1_CKPT, N_BANK, sha256, bank_vectors
import c1_pilot as C1
from strategy_channel_suite import build_greedy_env, CERT_TOWER, CERT_CONC
from ndcg_tree import build_pool

OUT = r"C:\Users\Tech\AppData\Local\Temp\claude\C--dev-phd\95869f84-0222-4b44-8a12-11de52045042\scratchpad\blend_coherence.json"

blob = torch.load(E1_CKPT, map_location="cpu")
assert sha256(E1_CKPT) == C1.C0_CKPT_SHA
E = build_greedy_env(CERT_TOWER, "validation", CERT_CONC, "paperC/coherence")
pool, _it, _cc = build_pool(E, 500, 500)
pool = [(int(a), int(b)) for a, b in pool]
assert pool == [(int(a), int(b)) for a, b in blob["pool"]]
pool_index = {q: i for i, q in enumerate(pool)}
full_pool = [(0, i) for i in range(E["ctx"].ni)] + [(1, c) for c in range(E["C"])]
assert len(full_pool) == N_BANK
norm = (blob["norm_mu"], blob["norm_sd"])
phi_t, _ = build_phi(E, pool, norm=norm)
phi_full, _ = build_phi(E, full_pool, norm=norm)
pol0 = Policy(E["ctx"].d, phi_full.shape[1], len(full_pool), arch=blob["arch"],
              state=blob["state"], use_pos=blob["use_pos"])
pol0.load_state_dict(blob["model"]); pol0.eval()
Ebank, Kbank = bank_vectors(pol0, phi_full)
Epool, Kpool = bank_vectors(pol0, phi_t)
log(f"[bank] Ebank {Ebank.shape} Epool {Epool.shape}")

Dva = load_traj(train=False)
reg, extra = C1.load_the_42(pool_index, Dva)
states = reg + extra
log(f"[states] {len(reg)} registered + {len(extra)} secondary")

# --- reproduce the pilot's targets exactly, as a gate -------------------------------------------
npz = np.load(os.path.join(PAPERC, "c1_pilot_per_state.npz"))
for m in (3, 5, 10):
    Tt, TKt, meta = C1.build_targets(states, Epool, Kpool, pool_index, m)
    d = float(np.abs(Tt - npz[f"target_m{m}"]).max())
    log(f"[gate] m={m} target reproduction max|diff| {d:.3g}")
    assert d < 1e-5, "the rebuilt targets are not the pilot's targets"

CH = np.asarray([c for c, _ in pool], np.int64)     # 0 = item, 1 = concept


def unit(X):
    return X / (np.linalg.norm(X, axis=-1, keepdims=True) + 1e-30)


def blend_stats(Ei, w):
    """Ei: (m, d) ingredient embeddings, w: (m,) weights summing to 1."""
    num = np.linalg.norm((w[:, None] * Ei).sum(0))
    den = float((w * np.linalg.norm(Ei, axis=1)).sum())
    C = unit(Ei) @ unit(Ei).T
    iu = np.triu_indices(len(w), 1)
    pw = C[iu]
    # weight-aware pairwise mean: sum_{i<j} w_i w_j cos / sum_{i<j} w_i w_j
    W = np.outer(w, w)[iu]
    return dict(ratio=float(num / (den + 1e-30)), pw_mean=float(pw.mean()),
                pw_min=float(pw.min()), pw_wmean=float((pw * W).sum() / W.sum()),
                pw_frac_neg=float((pw < 0).mean()))


def bank_mean_pairwise(X):
    U = unit(X.astype(np.float64))
    n = len(U)
    mu = U.mean(0)
    return float((n * float(mu @ mu) - 1.0) / (n - 1))


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

rng = np.random.default_rng(98765)
NDRAW = 200
res = {}
per_state = {}
for m in (3, 5, 10):
    _T, _TK, meta = C1.build_targets(states, Epool, Kpool, pool_index, m)
    for sname in ("homogeneous", "bias_free"):
        P = spaces[sname]["pool"]
        mu = spaces[sname]["mu"].astype(np.float32)
        Pc = P - mu
        rows = []
        for i, s in enumerate(states):
            q = np.asarray(meta[i]["top_m_pool_idx"], np.int64)
            w = np.asarray(meta[i]["weights"], np.float64)
            asked = {pool_index[a] for a in s["prefix"]}
            avail = np.asarray([j for j in range(len(pool)) if j not in asked], np.int64)
            raw = blend_stats(P[q].astype(np.float64), w)
            cen = blend_stats(Pc[q].astype(np.float64), w)
            rr, rc = [], []
            for _ in range(NDRAW):
                r = rng.choice(avail, size=m, replace=False)
                rr.append(blend_stats(P[r].astype(np.float64), w))
                rc.append(blend_stats(Pc[r].astype(np.float64), w))
            agg = lambda L, k: float(np.mean([x[k] for x in L]))
            rows.append(dict(sid=s["sid"], registered=bool(s.get("registered", False)),
                             n_concept=int((CH[q] == 1).sum()), n_item=int((CH[q] == 0).sum()),
                             raw=raw, centred=cen,
                             rand_raw={k: agg(rr, k) for k in raw},
                             rand_centred={k: agg(rc, k) for k in cen}))
        res[f"m{m}/{sname}"] = rows
        per_state[f"m{m}/{sname}"] = rows
        R = [r for r in rows if r["registered"]]
        log(f"[m={m} {sname}] n={len(R)} ratio med {np.median([r['raw']['ratio'] for r in R]):.4f} "
            f"| rand {np.median([r['rand_raw']['ratio'] for r in R]):.4f} "
            f"| pw_mean med {np.median([r['raw']['pw_mean'] for r in R]):.4f} "
            f"| centred ratio med {np.median([r['centred']['ratio'] for r in R]):.4f} "
            f"| centred pw med {np.median([r['centred']['pw_mean'] for r in R]):.4f}")

out = dict(bank={k: {kk: v[kk] for kk in ("bank_pw_raw", "bank_pw_centred", "mu_norm_ratio")}
                 for k, v in spaces.items()},
           n_random_draws=NDRAW, per_state=per_state)

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
        c["target_margin"] = dict(spearman=float(spearmanr(x, np.asarray(npz[f"target_m{m}_m"])[:42])[0]))
        if m == 5:
            for run in ("head_only_seed1", "full_seed1", "head_only_seed2", "full_seed2"):
                f = np.asarray(npz[f"cos_{run}"])
                c[f"fit_{run}"] = dict(pearson=float(pearsonr(x, f)[0]),
                                       spearman=float(spearmanr(x, f)[0]),
                                       p_spearman=float(spearmanr(x, f)[1]))
        # channel mix vs coherence
        nc = np.asarray([r["n_concept"] for r in rows])
        c["n_concept_vs_ratio"] = dict(spearman=float(spearmanr(nc, x)[0]),
                                       p_spearman=float(spearmanr(nc, x)[1]))
        c["mix_groups"] = {str(k): dict(n=int((nc == k).sum()),
                                        ratio_median=float(np.median(x[nc == k])) if (nc == k).any() else None,
                                        pw_median=float(np.median(np.asarray(
                                            [r["raw"]["pw_mean"] for r in rows])[nc == k])) if (nc == k).any() else None,
                                        centred_ratio_median=float(np.median(xc[nc == k])) if (nc == k).any() else None)
                           for k in range(0, m + 1)}
        corr[f"m{m}/{sname}"] = c
out["correlations"] = corr
json.dump(out, open(OUT, "w"), indent=1, default=float)
log(f"[done] wrote {OUT}")
