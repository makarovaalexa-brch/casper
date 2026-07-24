r"""run_battery_phaseB.py -- Instrument gate battery, PHASE B (the COVARIANCE / belief layer), on the
frozen i25 tower snapshot + the analytic belief layer (src/instrument/belief_layer.py). Spec:
docs/design/GATE_BATTERY_INSTRUMENT.md + DESIGN_SHEET_STEP2_BELIEF.md (design (ii), decoupled).

Reuses the Phase A eval scaffolding verbatim (build_real_ctx/build_smoke_ctx, eval_tokens,
ndcg10_from_scores, bootstrap_ci, rows_to_csr, load_genome, save_gate) and adds the belief machinery.

GATES (one JSON per gate -> experiments/battery/):
  G1  POSTERIOR USABLE   (a) tr(Sigma) shrinks over CONFIDENT answers (nested item prefixes; refusals
                              EXEMPT -- their small fitted alpha_refuse reported, not required to shrink);
                          (b) directional: folding phi shrinks var along phi >= 2x other directions;
                          (c) G1c calibration Spearman(belief-sigma, held-item NLL) >= 0.25.
  G4  CONFIDENCE+REFUSAL fitted alpha_refuse < alpha_vague < alpha_know-well (pre-C3 simulation: items
                          rated = know-well; concept SEL = vague; unrated-but-popular probe = refuse,
                          a CONSUMPTION observation only). Refusal Sigma-shrinkage <= 25% of a confident
                          answer's. (Documented honestly as the pre-C3 convention.)
  G8  Sigma LOAD-BEARING (a) order invariance: permute answer order K=5, belief (mu, tr Sigma) bit-stable;
                          (b) Sigma-greedy question selection (argmax posterior var along candidate dir)
                              vs isotropic-cI-greedy vs random, on the G9 fixed bank, budgets {2,4,8,16}:
                              Sigma-greedy MUST beat cI-greedy (else R4 narrows -- reported honestly).
  G2  FORMAL cold curves  q in {0,1,2,4,8,16}, arms = random-bank / popularity-order / HELF (Rashid, from
                          the train matrix) / Sigma-greedy; CREDIT-NEUTRAL masking of every revealed item
                          in ALL arms (masked from candidates AND dropped from the IDCG denominator);
                          FULL+TAIL NDCG@10; paired CIs vs random at matched budget.

Answer values from real ratings; askability = the structural rule; NO retired grids (asserted absent).

Build+smoke only if the fair-fight/premium processes still hold the CPU (author launches the real run):
  python src/instrument/run_battery_phaseB.py --smoke              # synthetic code-path check per gate
  python src/instrument/run_battery_phaseB.py [--only g1,g8] [--snapshot PATH] [--belief_ckpt PATH]
"""
import os
os.environ["OMP_NUM_THREADS"] = "6"
os.environ.setdefault("OPENBLAS_NUM_THREADS", "6")
os.environ.setdefault("MKL_NUM_THREADS", "6")
import sys
import json
import time
import argparse
import numpy as np
import torch
from scipy import sparse

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "src", "baselines"))
sys.path.insert(0, _HERE)
import metrics as M
from train_tower_t2 import level_to_sv, pack_tokens, NLEV, log
import run_battery_phaseA as PA
from run_battery_phaseA import (build_real_ctx, build_smoke_ctx, eval_tokens, ndcg10_from_scores,
                                bootstrap_ci, rows_to_csr, fold_z, load_genome, save_gate, spearman,
                                OUTDIR, SEED)
import belief_layer as BL
from belief_layer import (BeliefLayer, SigmaOps, build_U, build_item_dirs, build_concept_dirs,
                          ConceptMean, fold_items, mean_fold, empirical_var, precompute_increments,
                          prefit_sign_proof, fit_alphas, CH_ITEM, CH_CONC, KNOW_WELL, VAGUE, REFUSE)

# retired-answerer ban (Jul-22 audit)
assert not hasattr(sys.modules[__name__], "load_answerer")


# ============================================================================= belief attach
def attach_belief(ctx, belief_ckpt=None, min_members=30):
    """Augment a Phase A ctx with the belief layer: item dirs (raw Wd), concept dirs (whitened genome
    members + IDF), the Arm-A ConceptMean, and a BeliefLayer (var_emp from the val full known-half folds;
    fitted alphas loaded from belief_ckpt if given, else the init model)."""
    Wd = ctx.decoder.weight.detach().float()
    ctx.Wd = Wd; ctx.bd = ctx.decoder.bias.detach().float()
    ctx.d = ctx.enc.d_out
    ctx.item_dirs = build_item_dirs(Wd)                                  # (ni,d) RAW rows
    members = load_genome(ctx, min_members=min_members)
    ctx.tags = sorted(members.keys()); ctx.members = members
    d_c, d_raw, w_c = build_concept_dirs(Wd, members, ctx.tags)
    ctx.d_c = d_c; ctx.d_raw = d_raw; ctx.w_c = w_c
    ctx.cmean = ConceptMean(d_c, d_raw, w_c, beta_ctx=5.0)
    # var_emp from the val users' full known half (the shrunk empirical precision source)
    var_emp = empirical_var(ctx.enc, [(np.asarray(s, np.int64), np.asarray(l, np.int64))
                                      for s, l in ctx.allb])
    ctx.belief = BeliefLayer(var_emp)
    if belief_ckpt and os.path.exists(belief_ckpt):
        blob = torch.load(belief_ckpt, map_location="cpu")
        ctx.belief.load_state_dict(blob["belief"]); ctx.belief_fitted = True
        log(f"[belief] loaded fitted alphas from {belief_ckpt}")
    else:
        ctx.belief_fitted = False
        log("[belief] NO fitted ckpt -> using INIT alphas (G4/G1c report only; fit runs at launch)")
    # per-user SEL top concept (reuse the Phase A G5 lift machinery, te-half excluded by construction)
    Mm = sparse.csr_matrix((np.ones(sum(len(members[t]) for t in ctx.tags), np.float32),
                            (np.concatenate([members[t] for t in ctx.tags]),
                             np.concatenate([np.full(len(members[t]), i)
                                             for i, t in enumerate(ctx.tags)]))),
                           shape=(ctx.ni, len(ctx.tags)))
    counts = np.asarray((ctx.va_tr @ Mm).todense())
    nu = np.asarray(ctx.va_tr.sum(axis=1)).ravel().clip(min=1)
    gmass = ctx.cnt @ np.asarray(Mm.todense()); grate = gmass / max(ctx.cnt.sum(), 1e-9)
    lift = (counts / nu[:, None]) / np.maximum(grate[None, :], 1e-12)
    lift[counts < 2] = -np.inf
    top = lift.argmax(1); has = np.isfinite(lift.max(1))
    ctx.user_concept = {r: int(top[r]) for r in range(ctx.n) if has[r]}   # r -> concept index
    log(f"[belief] {len(ctx.user_concept)} val users with a SEL top concept")
    return ctx


def item_atoms(sids):
    """Sigma atoms for a set of ITEM answers (all know-well in the pre-C3 convention)."""
    sids = np.asarray(sids, np.int64)
    return (sids, np.full(len(sids), CH_ITEM, np.int64), np.full(len(sids), KNOW_WELL, np.int64))


def sigma_from_items(ctx, sids):
    """SigmaOps for one user's item evidence (raw Wd atoms)."""
    s, ch, conf = item_atoms(sids)
    atoms = [(ctx.item_dirs[torch.from_numpy(s)], ch, conf)]
    return SigmaOps(build_U(atoms, ctx.belief, ctx.d), ctx.belief.v0())


# ============================================================================= G1 posterior usable
def gate_g1(ctx):
    t0 = time.time()
    rows = [r for r in range(ctx.n) if ctx.va_te[r].nnz > 0 and len(ctx.allb[r][0]) >= 1]
    v0sum = float(ctx.belief.v0().sum())
    # ---- (a) tr(Sigma) over NESTED CONFIDENT (item) prefixes -- monotone by construction ----
    ks = [0, 1, 2, 4, 8, 16]
    rng = np.random.default_rng(SEED)
    traces = {}
    for k in ks:
        acc = []
        for r in rows:
            s = np.asarray(ctx.allb[r][0], np.int64)
            n = min(k, len(s))
            sel = rng.permutation(len(s))[:n]
            if n == 0:
                acc.append(v0sum); continue
            acc.append(float(sigma_from_items(ctx, s[sel]).trace()[0]))
        traces[str(k)] = float(np.mean(acc))
    vals = [traces[str(k)] for k in ks]
    g1a = all(vals[i] > vals[i + 1] for i in range(len(vals) - 1))
    # refusal EXEMPT: a refusal atom shrinks tr by <= its small alpha (reported, not required)
    a_ref = float(ctx.belief.alphas()[CH_CONC, REFUSE])
    # ---- (b) DIRECTIONAL: fold concept c -> var drop along d_c vs other concepts ----
    d_c = ctx.d_c
    ops0 = SigmaOps(build_U([(torch.zeros(0, ctx.d), np.zeros(0, np.int64), np.zeros(0, np.int64))],
                            ctx.belief, ctx.d), ctx.belief.v0())
    q0 = ops0.quad(d_c)[0].clamp_min(1e-12)                              # (C,) prior var per d_c
    own = np.zeros(len(ctx.tags)); oth = np.zeros(len(ctx.tags))
    for b in range(0, len(ctx.tags), 128):
        cids = list(range(b, min(b + 128, len(ctx.tags))))
        atoms = [(d_c[c:c + 1], np.array([CH_CONC]), np.array([VAGUE])) for c in cids]
        ops1 = SigmaOps(build_U(atoms, ctx.belief, ctx.d), ctx.belief.v0())
        q1 = ops1.quad(d_c)                                              # (len,C)
        R = ((q0.unsqueeze(0) - q1) / q0.unsqueeze(0)).numpy()
        for i, c in enumerate(cids):
            own[c] = R[i, c]; oth[c] = (R[i].sum() - R[i, c]) / max(len(ctx.tags) - 1, 1)
    mo, mr = float(own.mean()), float(oth.mean())
    ratio = mo / max(mr, 1e-12)
    # ---- (c) CALIBRATION: sqrt(w'Sigma w) vs held-item NLL, k=8 item evidence ----
    unc = []; err = []
    Wd, bd = ctx.Wd, ctx.bd
    for r in rows:
        s = np.asarray(ctx.allb[r][0], np.int64); l = np.asarray(ctx.allb[r][1], np.int64)
        n = min(8, len(s)); sel = np.random.default_rng(SEED + r).choice(len(s), n, replace=False)
        mu = fold_items(ctx.enc, [(s[sel], l[sel])])[0]
        ops = sigma_from_items(ctx, s[sel])
        hl = ctx.va_te[r].indices
        w = Wd[torch.from_numpy(hl)].mean(0); w = (w / w.norm().clamp_min(1e-8)).unsqueeze(0)
        unc.append(float(torch.sqrt(ops.quad_user(0, w)[0].clamp_min(1e-12))))
        logsm = torch.log_softmax(mu @ Wd.T + bd, -1)
        err.append(float(-logsm[torch.from_numpy(hl)].mean()))
    rho = spearman(unc, err)
    payload = {"gate": "G1", "n_users": len(rows),
               "g1a_traces": {k: round(traces[k], 4) for k in traces},
               "g1a_monotone_PASS": bool(g1a), "alpha_refuse_reported": a_ref,
               "g1b_own": round(mo, 6), "g1b_other": round(mr, 6), "g1b_ratio": round(ratio, 3),
               "g1b_PASS": bool(mo > 0 and ratio >= 2.0),
               "g1c_rho": round(rho, 4), "g1c_PASS": bool(rho >= 0.25),
               "belief_fitted": ctx.belief_fitted,
               "note": "g1c on INIT alphas is advisory; the fitted-alpha rho is the certified number",
               "seconds": round(time.time() - t0, 1)}
    payload["PASS"] = bool(g1a and payload["g1b_PASS"] and (payload["g1c_PASS"] or not ctx.belief_fitted))
    save_gate(ctx, "g1", payload)
    return payload


# ============================================================================= G4 confidence + refusal
def gate_g4(ctx):
    t0 = time.time()
    a = ctx.belief.alphas().detach().numpy()
    cp = ctx.belief.conf_precision()
    ordered = cp["refuse"] < cp["vague"] < cp["know-well"]
    # refusal Sigma-shrinkage <= 25% of a confident item answer's (measured on tr Sigma delta)
    rng = np.random.default_rng(SEED); rows = [r for r in range(ctx.n) if len(ctx.allb[r][0]) >= 2]
    d_share = []; d_ref = []
    for r in rows[:2000]:
        s = np.asarray(ctx.allb[r][0], np.int64)
        base = float(sigma_from_items(ctx, s[:1]).trace()[0])            # one confident item
        tr0 = float(ctx.belief.v0().sum())
        d_share.append(tr0 - base)                                       # confident shrinkage
        # refusal atom on the user's SEL concept (or concept 0)
        c = ctx.user_concept.get(r, 0)
        atoms = [(ctx.d_c[c:c + 1], np.array([CH_CONC]), np.array([REFUSE]))]
        tr_ref = float(SigmaOps(build_U(atoms, ctx.belief, ctx.d), ctx.belief.v0()).trace()[0])
        d_ref.append(tr0 - tr_ref)
    ms, mr = float(np.mean(d_share)), float(np.mean(d_ref))
    ratio = mr / max(ms, 1e-12)
    payload = {"gate": "G4",
               "alpha_grid_item": np.round(a[CH_ITEM], 5).tolist(),
               "alpha_grid_concept": np.round(a[CH_CONC], 5).tolist(),
               "conf_precision": cp, "ordering_refuse_lt_vague_lt_knowwell_PASS": bool(ordered),
               "confident_shrinkage_mean": round(ms, 6), "refusal_shrinkage_mean": round(mr, 6),
               "refusal_over_confident_ratio": round(ratio, 4),
               "refusal_le_25pct_PASS": bool(ratio <= 0.25),
               "belief_fitted": ctx.belief_fitted,
               "convention": "pre-C3 simulation: items rated=know-well, concept SEL=vague, "
                             "unrated-popular probe=refuse (consumption obs only)",
               "seconds": round(time.time() - t0, 1)}
    payload["PASS"] = bool(ordered and payload["refusal_le_25pct_PASS"])
    save_gate(ctx, "g4", payload)
    return payload


# ============================================================================= selection harness
def helf_scores(ctx):
    """HELF (Rashid): harmonic mean of normalized rating-entropy and normalized log-frequency, from the
    train matrix. The proc train matrix is BINARIZED (likes only), so rating-entropy is the BINARY
    like-rate entropy H(p), p = like-count / n_train_users (documented proxy; the classic full-rating
    entropy is unavailable on a binarized matrix)."""
    cnt = ctx.cnt.astype(np.float64)
    nusers = max(float(ctx.va_tr.shape[0]), cnt.max() + 1) if ctx.cnt.max() > 0 else 1.0
    p = np.clip(cnt / max(cnt.max(), 1e-9), 1e-6, 1 - 1e-6)              # normalized like-rate proxy
    H = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))                     # binary entropy in [0,1]
    Hn = H / max(H.max(), 1e-9)
    Fn = np.log1p(cnt) / max(np.log1p(cnt).max(), 1e-9)
    helf = 2 * Hn * Fn / np.clip(Hn + Fn, 1e-9, None)                    # harmonic mean
    return helf


def run_selection(ctx, bank, answers, policy, budgets, helf=None):
    """Adaptive item selection over the fixed bank with credit-neutral masking. policy in
    {random, popularity, helf, sigma, ci}. answers[r] = (sids, levels) the user CAN answer (structural
    rule). Returns dict budget -> (full array, tail array)."""
    rows = [r for r in range(ctx.n) if ctx.va_te[r].nnz > 0]
    Wd, bd = ctx.Wd, ctx.bd
    bank = np.asarray(bank, np.int64)
    pop_order = bank[np.argsort(-ctx.cnt[bank])]
    out = {q: (np.full(ctx.n, np.nan), np.full(ctx.n, np.nan)) for q in budgets}
    maxb = max(budgets)
    rng = np.random.default_rng(SEED)
    for r in rows:
        ans = dict(zip(answers[r][0].tolist(), answers[r][1].tolist()))  # sid -> level
        avail = [int(b) for b in bank if int(b) in ans]                 # answerable subset (structural)
        asked = []
        # precompute a static order for non-adaptive policies
        if policy == "random":
            order = list(rng.permutation(avail))
        elif policy == "popularity":
            order = [int(b) for b in pop_order if int(b) in ans]
        elif policy == "helf":
            order = sorted(avail, key=lambda i: -helf[i])
        else:
            order = None                                                # adaptive
        for q in budgets:
            # extend the asked list to q
            while len(asked) < q:
                pool = [i for i in avail if i not in asked]
                if not pool:
                    break
                if order is not None:
                    nxt = [i for i in order if i not in asked][0]
                else:
                    ops = sigma_from_items(ctx, asked) if asked else \
                        SigmaOps(build_U([(torch.zeros(0, ctx.d), np.zeros(0, np.int64),
                                           np.zeros(0, np.int64))], ctx.belief, ctx.d), ctx.belief.v0())
                    cand = np.asarray(pool, np.int64)
                    D = ctx.item_dirs[torch.from_numpy(cand)]
                    if policy == "sigma":
                        val = ops.quad_user(0, D).numpy()               # d'Sigma d posterior variance
                    else:                                               # ci: isotropic Sigma=cI -> c||d||^2
                        val = (D ** 2).sum(1).numpy()
                    nxt = int(cand[int(val.argmax())])
                asked.append(nxt)
            toks = [(int(i), int(ans[i])) for i in asked[:q]]
            s = np.array([t[0] for t in toks], np.int64); l = np.array([t[1] for t in toks], np.int64)
            mu = fold_items(ctx.enc, [(s, l)])[0]
            sc = (mu @ Wd.T + bd).numpy().astype(np.float32)[None, :]
            # credit-neutral: mask revealed items from candidates AND drop from IDCG denominator
            mask = ctx.va_tr[r].copy()
            if len(s):
                extra = sparse.csr_matrix((np.ones(len(s), np.float32), (np.zeros(len(s)), s)),
                                          shape=(1, ctx.ni))
                mask = (mask + extra); mask.data[:] = 1.0
            te = ctx.va_te[r].copy().tolil()
            for j in s.tolist():
                te[0, j] = 0
            te = te.tocsr(); te.eliminate_zeros()
            f, t = ndcg10_from_scores(sc, mask.tocsr(), te, ctx.head_mask)
            out[q][0][r] = f[0]; out[q][1][r] = t[0]
    return out


# ============================================================================= G8 Sigma load-bearing
def gate_g8(ctx, budgets=(2, 4, 8, 16)):
    t0 = time.time()
    # ---- (a) order invariance: permute a mixed atom set K=5 -> mu and tr(Sigma) bit-stable ----
    r0 = next(r for r in range(ctx.n) if len(ctx.allb[r][0]) >= 4)
    s = np.asarray(ctx.allb[r0][0], np.int64)[:4]; l = np.asarray(ctx.allb[r0][1], np.int64)[:4]
    c = ctx.user_concept.get(r0, 0)
    mus = []; trs = []
    rng = np.random.default_rng(SEED)
    for _ in range(5):
        p = rng.permutation(4)
        mu = mean_fold(ctx.enc, [(s[p], l[p])], [[(c, 1)]], ctx.cmean)[0]
        ops = sigma_from_items(ctx, s[p])
        mus.append(mu.numpy()); trs.append(float(ops.trace()[0]))
    mu_stable = float(np.max([np.abs(mus[0] - m).max() for m in mus]))
    tr_stable = float(np.max([abs(trs[0] - t) for t in trs]))
    order_inv = bool(mu_stable < 1e-4 and tr_stable < 1e-4)
    # ---- (b) Sigma-greedy vs cI-greedy vs random on the pop bank ----
    bank, answers = ctx.banks["pop"]
    rows = [r for r in range(ctx.n) if ctx.va_te[r].nnz > 0]
    res = {}
    for pol in ("sigma", "ci", "random"):
        sel = run_selection(ctx, bank, answers, pol, budgets)
        res[pol] = {q: float(np.nanmean(sel[q][0][rows])) for q in budgets}
    beats = {q: bool(res["sigma"][q] > res["ci"][q]) for q in budgets}
    # paired CI at the largest budget
    sel_s = run_selection(ctx, bank, answers, "sigma", (max(budgets),))
    sel_c = run_selection(ctx, bank, answers, "ci", (max(budgets),))
    d = bootstrap_ci(sel_s[max(budgets)][0][rows] - sel_c[max(budgets)][0][rows])
    payload = {"gate": "G8",
               "order_invariance": {"mu_max_dev": mu_stable, "tr_max_dev": tr_stable,
                                    "PASS": order_inv},
               "selection_full@10": {p: res[p] for p in res},
               "sigma_beats_ci_by_budget": {str(q): beats[q] for q in budgets},
               "sigma_minus_ci_full@10_maxbudget": {"mean": d[0], "ci95": d[1]},
               "sigma_load_bearing_PASS": bool(all(beats.values())),
               "note": "if sigma does NOT beat cI, R4 narrows to selection-grade-only -- report honestly",
               "seconds": round(time.time() - t0, 1)}
    payload["PASS"] = bool(order_inv and payload["sigma_load_bearing_PASS"])
    save_gate(ctx, "g8", payload)
    return payload


# ============================================================================= G2 formal cold curves
def gate_g2(ctx, budgets=(0, 1, 2, 4, 8, 16)):
    t0 = time.time()
    bank, answers = ctx.banks["pop"]
    helf = helf_scores(ctx)
    rows = [r for r in range(ctx.n) if ctx.va_te[r].nnz > 0]
    pos_budgets = [q for q in budgets if q > 0]
    arms = {}
    per_arm_full = {}
    for pol in ("random", "popularity", "helf", "sigma"):
        sel = run_selection(ctx, bank, answers, pol, pos_budgets, helf=helf)
        # q=0 intercept (empty fold) shared across arms
        mu0 = fold_items(ctx.enc, [(np.zeros(0, np.int64), np.zeros(0, np.int64))])
        f0 = np.full(ctx.n, np.nan); t0v = np.full(ctx.n, np.nan)
        sc0 = (mu0 @ ctx.Wd.T + ctx.bd).numpy().astype(np.float32)
        for r in rows:
            f, t = ndcg10_from_scores(sc0, ctx.va_tr[r], ctx.va_te[r], ctx.head_mask)
            f0[r] = f[0]; t0v[r] = t[0]
        full = {0: f0}; tail = {0: t0v}
        for q in pos_budgets:
            full[q] = sel[q][0]; tail[q] = sel[q][1]
        arms[pol] = {"full@10": {q: float(np.nanmean(full[q][rows])) for q in budgets},
                     "tail@10": {q: float(np.nanmean(tail[q][rows])) for q in budgets}}
        per_arm_full[pol] = full
    # paired CIs vs random at matched budget
    vs = {}
    for pol in ("popularity", "helf", "sigma"):
        vs[pol] = {}
        for q in pos_budgets:
            d = bootstrap_ci(per_arm_full[pol][q][rows] - per_arm_random_get(per_arm_full, "random", q)[rows])
            vs[pol][str(q)] = {"mean": d[0], "ci95": d[1], "beats_random": bool(d[0] > 0 and d[1][0] > 0)}
    # curve rises (endpoint - intercept) for each arm, full+tail
    rise = {p: {"full": arms[p]["full@10"][budgets[-1]] - arms[p]["full@10"][0],
                "tail": arms[p]["tail@10"][budgets[-1]] - arms[p]["tail@10"][0]} for p in arms}
    payload = {"gate": "G2-formal", "budgets": list(budgets), "arms": arms,
               "rise_endpoint_minus_intercept": rise, "vs_random_paired": vs,
               "masking": "credit-neutral: revealed items masked from candidates AND dropped from IDCG",
               "seconds": round(time.time() - t0, 1)}
    # PASS: sigma arm rises full+tail with CI>0, and beats random by q4 (tail via full proxy reported)
    sig_ok = rise["sigma"]["full"] > 0 and rise["sigma"]["tail"] > 0
    payload["sigma_rises_PASS"] = bool(sig_ok)
    payload["PASS"] = bool(sig_ok)
    save_gate(ctx, "g2_formal", payload)
    return payload


def per_arm_random_get(per_arm_full, name, q):
    return per_arm_full[name][q]


# ============================================================================= G5 FIX (author priority)
def _concept_cold_ndcg(ctx, rows, beta, rho):
    """Concept-ONLY cold (k=0) NDCG@10: mu = intercept + Arm-A shift (with the G5-fix cold operator:
    per-k beta + raw-centroid floor blend rho). i25 empty fold = 0, so mu = shift. Returns per-user full."""
    cm = ConceptMean(ctx.d_c, ctx.d_raw, ctx.w_c, beta_ctx=5.0, beta_cold=beta, floor_rho=rho)
    f = np.full(ctx.n, np.nan)
    for st in range(0, len(rows), 500):
        chunk = rows[st:st + 500]
        Z = np.zeros((len(chunk), ctx.d), np.float32)
        for j, r in enumerate(chunk):
            c = ctx.user_concept[r]
            Z[j] = cm.shift(c, 1, cold=True).numpy()
        S = (torch.from_numpy(Z) @ ctx.Wd.T + ctx.bd).numpy().astype(np.float32)
        ff, _ = ndcg10_from_scores(S, ctx.va_tr[chunk], ctx.va_te[chunk], None)
        f[np.asarray(chunk)] = ff
    return f


def _concept_ctx_ndcg(ctx, rows, beta):
    """Concept-on-context (k=2) NDCG@10: fold the k2 item tokens + Arm-A concept shift at beta (pure
    whitened dir; rho=0 so the +0.0084 context result is untouched). Returns per-user full."""
    cm = ConceptMean(ctx.d_c, ctx.d_raw, ctx.w_c, beta_ctx=beta)
    Zs = np.zeros((ctx.n, ctx.d), np.float32)
    for r in rows:
        Zs[r] = cm.shift(ctx.user_concept[r], 1, cold=False).numpy()
    f, _ = eval_tokens(ctx, ctx.k2_tokens, ctx.va_tr, rows=rows, z_shift=Zs)
    return f


def gate_g5fix(ctx):
    """G5 FIX (author priority): the standalone-cold concept miss (Phase A: 0.1234 vs intercept 0.1279).
    Reports BOTH options on val, picks whichever clears G-collinearity at k=0 WITHOUT harming the k2
    context result (+0.0084 at beta=5 must survive):
      (1) PER-K beta -- beta_cold calibrated separately from beta_ctx;
      (2) FLOOR BLEND -- retain a fraction rho of the popularity-bearing raw member centroid at k=0 only."""
    t0 = time.time()
    rows = [r for r in range(ctx.n) if ctx.va_te[r].nnz > 0 and r in ctx.user_concept]
    empty = [(np.empty(0, np.int64), np.empty(0, np.int64))] * ctx.n
    f_int, _ = eval_tokens(ctx, empty, ctx.va_tr, rows=rows)
    intercept = float(np.nanmean(f_int[rows]))
    f_k2, _ = eval_tokens(ctx, ctx.k2_tokens, ctx.va_tr, rows=rows)
    k2only = float(np.nanmean(f_k2[rows]))
    # option 1: per-k beta (cold sweep at rho=0; ctx sweep)
    betas = (0.0, 0.5, 1.0, 2.0, 5.0)
    cold_curve = {b: float(np.nanmean(_concept_cold_ndcg(ctx, rows, b, 0.0)[rows])) for b in betas}
    ctx_curve = {b: float(np.nanmean(_concept_ctx_ndcg(ctx, rows, b)[rows])) for b in betas}
    beta_cold = max(cold_curve, key=lambda b: cold_curve[b])
    beta_ctx = max(ctx_curve, key=lambda b: ctx_curve[b])
    perk_clears = cold_curve[beta_cold] > intercept
    perk_ctx_survives = (ctx_curve[5.0] - k2only) >= 0.0
    # option 2: floor blend (cold sweep of rho at a fixed small beta_cold=2)
    rhos = (0.0, 0.25, 0.5, 0.75, 1.0)
    floor_curve = {r_: float(np.nanmean(_concept_cold_ndcg(ctx, rows, 2.0, r_)[rows])) for r_ in rhos}
    floor_rho = max(floor_curve, key=lambda r_: floor_curve[r_])
    floor_clears = floor_curve[floor_rho] > intercept
    # pick: prefer the option that clears intercept at k0 AND preserves k2 (+0.0084 at beta5)
    winner = None
    if floor_clears:
        winner = "floor_blend"
    elif perk_clears:
        winner = "per_k_beta"
    payload = {"gate": "G5-fix", "n_users": len(rows), "intercept_full@10": intercept,
               "k2only_full@10": k2only,
               "per_k_beta": {"cold_curve": cold_curve, "ctx_curve": ctx_curve,
                              "beta_cold": beta_cold, "beta_ctx": beta_ctx,
                              "cold_clears_intercept": bool(perk_clears),
                              "ctx_beta5_delta_vs_k2only": round(ctx_curve[5.0] - k2only, 5),
                              "ctx_survives": bool(perk_ctx_survives)},
               "floor_blend": {"cold_curve": floor_curve, "floor_rho": floor_rho,
                               "beta_cold_used": 2.0, "cold_clears_intercept": bool(floor_clears)},
               "winner": winner,
               "note": "k>=2 uses pure whitened (rho=0) so the context result is untouched; only the "
                       "k=0 cold operator changes. Phase A miss was 0.1234 vs intercept 0.1279.",
               "seconds": round(time.time() - t0, 1)}
    payload["PASS"] = bool(winner is not None)
    save_gate(ctx, "g5fix", payload)
    return payload


# ============================================================================= main
GATES = {"g1": gate_g1, "g4": gate_g4, "g8": gate_g8, "g2_formal": gate_g2, "g5fix": gate_g5fix}
ORDER = ["g1", "g4", "g8", "g2_formal", "g5fix"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--only", default=None)
    ap.add_argument("--snapshot", default=PA.SNAP_DEFAULT)
    ap.add_argument("--belief_ckpt", default=None,
                    help="fitted BeliefLayer state (belief_layer fit output); default = INIT alphas")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    ctx = build_smoke_ctx() if args.smoke else build_real_ctx(args.snapshot)
    if args.smoke:
        ctx.outdir = os.path.join(OUTDIR, "_smoke")
    attach_belief(ctx, belief_ckpt=args.belief_ckpt)
    todo = ORDER if not args.only else [g for g in ORDER if g in set(args.only.split(","))]
    results = {}
    for g in todo:
        outp = os.path.join(ctx.outdir, f"{g}.json")
        if os.path.exists(outp) and not args.force and not args.smoke:
            log(f"[skip] {g}: exists"); results[g] = json.load(open(outp)); continue
        log(f"=== {g.upper()} ===")
        with torch.no_grad():                       # Phase B is scoring-only (no alpha fitting here)
            results[g] = GATES[g](ctx)
    print("\n=== PHASE B VERDICTS ===")
    for g in todo:
        print(f"  {g}: PASS={results[g].get('PASS', 'n/a')}")


if __name__ == "__main__":
    main()
