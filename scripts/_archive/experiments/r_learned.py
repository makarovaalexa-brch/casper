"""r_learned.py -- THE LEARNED BELIEF-COMBINER ROUTER (author-specified).

Value-belief and vividness-belief enter as FEATURES; the COMBINATION is LEARNED from NDCG outcome
labels; NOTHING optimizes for answerability per se. The learner sees only population beliefs +
blind state; it predicts 1-step realized NDCG@10 gain; the policy asks argmax predicted gain.

  r-learned-free     : pure argmax over candidates (full expressiveness).
  r-learned-anchored : tie-by-construction on the SPLIT-FAIR static -- swap only when the learner's
                       predicted gain exceeds the scheduled question's by a val-fitted margin
                       (margin fit on POPULATION val, firewall-clean).

FAIR HARNESS (E1-E7, STATE_2026-07-08):
  - four split-estimates: seeds {0,1} x eval-half {A,B}; all users of the eval half in every arm.
  - baseline = the SPLIT-CONSTRUCTED s-item / s-mixed statics (greedy built on the OTHER half),
    reusing scripts/static_contamination.py machinery (E1, de-contaminated per STATIC_CONTAMINATION).
  - T=12 primary (+8/24 columns); refusal = no-op turn (E1); paired per-user bootstrap.
  - ARM-SYMMETRY TABLE (E7): static learns outcomes from the CONSTRUCTION-HALF study users;
    r-learned learns outcomes ONLY from trU population sims (strictly LESS arena information).

PRE-REGISTERED READS (printed before results):
  (i)   r-learned(-anchored) vs s-fair pooled CI -- positive excl 0 = first fair learned-adaptivity win.
  (ii)  feature importances: does the learner USE the vividness belief (importance > noise floor)?
  (iii) r-learned-free must not lose to r-learned-anchored by much (expressiveness check).
  (iv)  never lose to s-fair beyond noise (E2).

NO LLM calls; all local. DIRECTIONAL 173/300 (grid unfrozen) -- do not cite.
Run:  python scripts/r_learned.py --do all   (train needs .cache/r_learned_labels/*.npz)
"""
import os, sys, json, time, argparse, glob, collections
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import warnings
warnings.filterwarnings("ignore")
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
import repair_probes as RP
import static_contamination as SC
import i25_fold_v2 as V2
import i25_phase4_fair as FA
import i25_lib as L
import r_learned_labels as RL

MODEL_CACHE = ".cache/r_learned_model.pkl"
RESULT_JSON = ".cache/r_learned_results.json"
OUT_MD = "experiments/R_LEARNED.md"
BOOT = 5000
BUDGETS = [8, 12, 24]
TMAX = max(BUDGETS)
PRIMARY_BUDGET = 12
MARGIN_GRID = [0.0, 0.001, 0.002, 0.005, 0.01, 0.02]


def md(txt, mode="a"):
    os.makedirs("experiments", exist_ok=True)
    open(OUT_MD, mode, encoding="utf-8").write(txt)


# ==================================================================== load label dataset
def load_labels(seed=0):
    files = sorted(glob.glob(os.path.join(RL.LAB_DIR, f"chunk_{seed}_*.npz")))
    if not files:
        raise SystemExit(f"no label chunks in {RL.LAB_DIR} for seed {seed}; run r_learned_labels.py")
    X, Y, U, Tn, Ch = [], [], [], [], []
    for f in files:
        z = np.load(f)
        X.append(z["X"]); Y.append(z["Y"]); U.append(z["user"]); Tn.append(z["turn"]); Ch.append(z["ch"])
    X = np.concatenate(X); Y = np.concatenate(Y); U = np.concatenate(U)
    Tn = np.concatenate(Tn); Ch = np.concatenate(Ch)
    print(f"[labels] {len(X)} examples from {len(files)} chunks; {len(np.unique(U))} users", flush=True)
    return X, Y, U, Tn, Ch


# ==================================================================== train the combiner + importances
def train(seed=0):
    import pickle
    X, Y, U, Tn, Ch = load_labels(seed)
    uids = np.unique(U)
    rng = np.random.default_rng(0)
    val_u = set(uids[rng.permutation(len(uids))[:max(1, len(uids) // 5)]].tolist())
    is_val = np.array([u in val_u for u in U])
    Xtr, Ytr = X[~is_val], Y[~is_val]
    Xva, Yva = X[is_val], Y[is_val]
    print(f"[train] train {len(Xtr)} / val {len(Xva)} (val split by USER)", flush=True)
    gbm = HistGradientBoostingRegressor(max_iter=400, max_depth=None, max_leaf_nodes=31,
                                        learning_rate=0.05, l2_regularization=1.0,
                                        min_samples_leaf=50, early_stopping=False, random_state=seed)
    gbm.fit(Xtr, Ytr)
    tr_r2 = gbm.score(Xtr, Ytr); va_r2 = gbm.score(Xva, Yva)
    print(f"[train] R^2 train {tr_r2:.4f} val {va_r2:.4f}", flush=True)

    # permutation importance on a val subsample (noise_floor feature = the reference floor)
    idx = rng.permutation(len(Xva))[:min(20000, len(Xva))]
    pim = permutation_importance(gbm, Xva[idx], Yva[idx], n_repeats=8, random_state=seed,
                                 scoring="r2")
    imp = pim.importances_mean; imp_sd = pim.importances_std
    feat_imp = {RL.FEAT_NAMES[i]: float(imp[i]) for i in range(RL.NF)}
    feat_imp_sd = {RL.FEAT_NAMES[i]: float(imp_sd[i]) for i in range(RL.NF)}
    noise_floor = feat_imp["noise_floor"]
    grp = {}
    for g, names in RL.FEAT_GROUPS.items():
        grp[g] = float(sum(feat_imp[n] for n in names))
    order = sorted(feat_imp.items(), key=lambda kv: -kv[1])
    print("[importance] per-feature (permutation, r2 drop):", flush=True)
    for n, v in order:
        flag = "  <-- NOISE FLOOR" if n == "noise_floor" else (
            "  (> floor)" if v > noise_floor + feat_imp_sd.get("noise_floor", 0) else "")
        print(f"    {n:24s} {v:+.5f} +/- {feat_imp_sd[n]:.5f}{flag}", flush=True)
    print("[importance] grouped:", flush=True)
    for g, v in sorted(grp.items(), key=lambda kv: -kv[1]):
        print(f"    {g:16s} {v:+.5f}", flush=True)

    blob = dict(model=gbm, feat_imp=feat_imp, feat_imp_sd=feat_imp_sd, grouped=grp,
                noise_floor=noise_floor, tr_r2=float(tr_r2), va_r2=float(va_r2),
                n_examples=int(len(X)), n_users=int(len(uids)), feat_names=RL.FEAT_NAMES)
    pickle.dump(blob, open(MODEL_CACHE, "wb"))
    print(f"[train] saved model + importances -> {MODEL_CACHE}", flush=True)
    return blob


def load_model():
    import pickle
    return pickle.load(open(MODEL_CACHE, "rb"))


# ==================================================================== vectorized feature matrix (eval)
class FeatBuilder:
    """Precompute z-independent feature columns for the candidate universe; per (z,t,ac) fills the
    taste-dependent columns. Numerically identical to RL.make_features (asserted in smoke)."""
    def __init__(self, CANDS, tables):
        self.CANDS = CANDS; self.tables = tables
        self.EMB = np.stack([m["emb"] for m in CANDS]).astype(np.float64)   # (nc,d)
        self.EMBN = np.linalg.norm(self.EMB, axis=1) + 1e-9
        self.typ = np.array([m["typ"] for m in CANDS])
        self.popv = tables["pop_value"]; self.ans = tables["pop_answerability"]
        self.viv = tables["pop_vividness"]; self.tier = tables["value_tier"]

    def matrix(self, cand_ids, z, t, ac, noise_vec):
        c = np.asarray(cand_ids)
        emb = self.EMB[c]; nz = float(np.linalg.norm(z))
        taste = (emb @ z) / (nz * self.EMBN[c]) if nz > 0 else np.zeros(len(c))
        tgate = 0.5 * (taste + 1.0)
        m = len(c); F = np.zeros((m, RL.NF))
        F[:, 0] = (self.typ[c] == 0); F[:, 1] = (self.typ[c] == 1); F[:, 2] = (self.typ[c] == 2)
        F[:, 3] = self.popv[c]; F[:, 4] = self.ans[c]; F[:, 5] = self.viv[c]
        F[:, 6] = taste; F[:, 7] = taste * (t / RL.TMAX)
        F[:, 8] = self.ans[c] * tgate; F[:, 9] = self.viv[c] * tgate
        F[:, 10] = self.tier[c]; F[:, 11] = t; F[:, 12] = nz; F[:, 13] = ac
        F[:, 14] = noise_vec
        return F


# ==================================================================== adaptive router evaluation
def eval_router(env, tables, gbm, sub_users, sub_cold, T, mode="free", s_fair=None, margin=0.0,
                answer_fn=None):
    """Per-turn batched adaptive rollout. answer_fn(i,rec,cd) -> (answered, token, native_or_None).
    Returns per-user per-turn NDCG@10 (n,T). Refusal = no-op (belief unchanged)."""
    FR, model, CANDS = env["FR"], env["model"], env["CANDS"]
    n = len(sub_users); nc = len(CANDS)
    FB = FeatBuilder(CANDS, tables)
    used = [set() for _ in range(n)]
    ev_tok = [[] for _ in range(n)]; ev_nat = [[] for _ in range(n)]
    ac = np.zeros(n, int); z_cur = [np.zeros(FR.W.shape[1]) for _ in range(n)]
    per_turn = np.empty((n, T)); per_turn.fill(np.nan)
    nrng = np.random.default_rng(777)
    for t in range(T):
        chosen = np.empty(n, int)
        for i in range(n):
            cand = [c for c in range(nc) if c not in used[i]]
            F = FB.matrix(cand, z_cur[i], t, ac[i], nrng.random(len(cand)))
            pred = gbm.predict(F)
            if mode == "free" or s_fair is None:
                bc = cand[int(np.argmax(pred))]
            else:
                sched_c = next((c for c in s_fair if c not in used[i]), None)
                best_c = cand[int(np.argmax(pred))]
                if sched_c is None:
                    bc = best_c
                else:
                    pg = dict(zip(cand, pred))
                    bc = best_c if (pg[best_c] - pg.get(sched_c, -1e18)) > margin else sched_c
            chosen[i] = bc; used[i].add(bc)
        # observe answers + fold
        for i in range(n):
            answered, tok, nat = answer_fn(i, sub_users[i], chosen[i])
            if answered:
                ev_tok[i].append(tok)
                if nat is not None:
                    ev_nat[i].append(nat)
                ac[i] += 1
        ztl, znl, zidx = [], [], []
        for i in range(n):
            if ev_tok[i]:
                ztl.append(ev_tok[i]); znl.append(ev_nat[i]); zidx.append(i)
        col = sub_cold.copy()
        if ztl:
            Z = V2.fold_batch_v2(FR, model, ztl, znl)
            vals = FA.ndcg_batch(FR, Z, [sub_users[i] for i in zidx], 10)
            for r, i in enumerate(zidx):
                col[i] = vals[r]; z_cur[i] = Z[r]
        per_turn[:, t] = col
    return per_turn


def study_answer_fn(env):
    CANDS = env["CANDS"]
    def fn(i, rec, cd):
        if rec["ans_arr"][cd]:
            return True, RP.tok_of(CANDS, rec, cd), rec["nat_arr"][cd]
        return False, None, None
    return fn


def paired(a, b, seed=0):
    a = np.asarray(a, float); b = np.asarray(b, float)
    d = a - b; rng = np.random.default_rng(seed)
    bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(BOOT)])
    return dict(delta=float(d.mean()), ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                n=int(len(d)), a=float(a.mean()), b=float(b.mean()))


# ==================================================================== margin fit (POPULATION val)
def fit_margin(env, tables, gbm, seed=0, n_pop=400):
    """Fit the anchored swap margin on POPULATION val users (firewall). Population anchor = the
    pop_value-ranked static (a legitimate population static, no study data). Pick the margin
    maximizing population anytime NDCG@12."""
    FR, model, CANDS = env["FR"], env["model"], env["CANDS"]
    P = RL.pop_env()
    prof = L.load_train_profiles(P["D"], n_pop, seed=seed + 7000)
    rng = np.random.default_rng(seed + 7001)
    pop_users, sim = [], []
    for u, p in prof.items():
        known, held = V2.make_user_split(p, rng)
        if held and len(known) >= 4:
            a = RL.simulate_answers(P, known, np.random.default_rng(seed * 13 + int(u)))
            if len(a) >= 2:
                pop_users.append(dict(u=u, tlike=held, prof=set(known.keys()), sim=a))
                sim.append(a)
    if not pop_users:
        return 0.0, {}
    cold = FA.ndcg_batch(FR, np.zeros((len(pop_users), FR.W.shape[1]), np.float32), pop_users, 10)
    # population anchor: pop_value-ranked schedule (top TMAX by pop_value)
    pop_rank = list(np.argsort(-tables["pop_value"]))
    s_pop = pop_rank[:TMAX]

    def afn(i, rec, cd):
        if cd in rec["sim"]:
            tok, nat = rec["sim"][cd]
            return True, tok, nat
        return False, None, None

    scores = {}
    for mg in MARGIN_GRID:
        pt = eval_router(env, tables, gbm, pop_users, cold, PRIMARY_BUDGET, mode="anchored",
                         s_fair=s_pop, margin=mg, answer_fn=afn)
        scores[mg] = float(np.nanmean(pt[:, :PRIMARY_BUDGET].mean(axis=1)))
        print(f"  [margin {mg:.3f}] pop val anytime@12 = {scores[mg]:.4f}", flush=True)
    best = max(scores, key=lambda m: scores[m])
    print(f"  [margin] fitted margin = {best} (pop val {scores[best]:.4f})", flush=True)
    return best, scores


# ==================================================================== the fair evaluation
PREREG = (
    "## Pre-registered reads (printed BEFORE results)\n\n"
    "Verdicts use the POOLED estimate (all 4 split-estimates' per-user deltas concatenated), T=12.\n\n"
    "- **(i) FIRST FAIR LEARNED-ADAPTIVITY WIN:** r-learned-anchored vs s-fair pooled CI excludes 0\n"
    "  and is POSITIVE -> the learned belief-combiner beats the split-constructed static under strictly\n"
    "  less arena information.\n"
    "- **(ii) DOES THE LEARNER USE VIVIDNESS?** the VIVIDNESS group's permutation importance exceeds\n"
    "  the noise-floor feature's importance (+1 sd) -> the learner weights vividness once value is\n"
    "  controlled (the author's hypothesis, made measurable). Reported for VALUE / VIVIDNESS /\n"
    "  ANSWERABILITY side by side.\n"
    "- **(iii) EXPRESSIVENESS:** r-learned-free does not lose to r-learned-anchored beyond noise.\n"
    "- **(iv) E2 FLOOR:** neither router loses to s-fair beyond noise (pooled CI lower bound >= -0.01).\n\n"
)

SYMMETRY = (
    "## ARM-SYMMETRY TABLE (E7 -- what each arm sees / learns from / is constructed on)\n\n"
    "| arm | selects using | learns outcomes from | constructed on | sees study held-out targets? |\n"
    "|---|---|---|---|---|\n"
    "| s-item / s-mixed (SPLIT-FAIR) | greedy schedule (fixed) | cohort-mean NDCG on the CONSTRUCTION-HALF study users | the OTHER study half (out-of-sample) | YES at construction (other half's targets) |\n"
    "| r-learned-free | learned gain model on population features + blind state z | 1-step NDCG gain on trU POPULATION sims ONLY | trU population (firewall) | NO -- never |\n"
    "| r-learned-anchored | s-fair order, swap on learned-gain margin | trU POPULATION sims (model) + POPULATION val (margin) | s-fair schedule + population | NO -- never |\n\n"
    "> ASYMMETRY (stated): the static is BUILT against study-user outcomes (its construction half's\n"
    "> held-out targets); r-learned's combiner and margin are BUILT only on trU population simulations.\n"
    "> The router therefore competes with STRICTLY LESS arena information. A win is conservative; a tie\n"
    "> is the price of the firewall; a loss beyond noise (E2) triggers diagnosis.\n\n"
)


def evaluate(margin=None):
    t0 = time.time()
    env = SC.RP.setup(); env = SC.RP.value_tiers(env)
    RL.pop_env()                         # ensures CANDS carry members/base_kw (shared list)
    tables = RL.build_feature_tables(RL.pop_env(), n_users=4000, seed=0)
    blob = load_model(); gbm = blob["model"]
    users = env["users"]; n = len(users); CANDS = env["CANDS"]
    item_pool = [m["cid"] for m in CANDS if m["kind"] == "item"]
    mixed_pool = [m["cid"] for m in CANDS]
    afn = study_answer_fn(env)

    md("# r-learned -- THE LEARNED BELIEF-COMBINER ROUTER (DIRECTIONAL 173/300)\n\n", mode="w")
    md("> **DIRECTIONAL BANNER** -- 173/300 users, answerer-v1 working grid NOT frozen; v2 fold\n"
       "> `.cache/i25_fold_v2_best.pt`. Re-run on the frozen 300-user grid before any citation.\n\n"
       f"Date 2026-07-09. Scripts `scripts/r_learned.py` (+ `scripts/r_learned_labels.py`). NO LLM\n"
       f"calls; local compute. {n} study users; {len(CANDS)} candidates; cold NDCG@10 "
       f"{env['cold'].mean():.4f}; fold val {env['foldval']:.4f}. Paired per-user bootstrap BOOT={BOOT}.\n\n"
       f"Learner: HistGradientBoosting on {blob['n_examples']} population (state,candidate,label)\n"
       f"examples from {blob['n_users']} trU users; val R^2 {blob['va_r2']:.4f}. Label = 1-step realized\n"
       f"NDCG@10 gain under the certified fold-v2 sampler. Features = SELECTION-TIME population beliefs\n"
       f"only (see r_learned_labels.py header).\n\n")
    md(SYMMETRY); md(PREREG)
    print(SYMMETRY, flush=True); print(PREREG, flush=True)

    # ---- feature importances (read ii) ----
    md("## Feature importances (read ii -- value vs vividness vs answerability)\n\n"
       f"Permutation importance (r2 drop) on the by-user val split; noise-floor feature = the "
       f"reference floor.\n\n| feature | importance | +/- sd | > floor? |\n|---|--:|--:|---|\n")
    nf = blob["noise_floor"]; nfsd = blob["feat_imp_sd"]["noise_floor"]
    for name, v in sorted(blob["feat_imp"].items(), key=lambda kv: -kv[1]):
        sd = blob["feat_imp_sd"][name]
        above = "yes" if (name != "noise_floor" and v > nf + nfsd) else ("FLOOR" if name == "noise_floor" else "no")
        md(f"| {name} | {v:+.5f} | {sd:.5f} | {above} |\n")
    md("\n**Grouped importance:**\n\n| group | importance |\n|---|--:|\n")
    for g, v in sorted(blob["grouped"].items(), key=lambda kv: -kv[1]):
        md(f"| {g} | {v:+.5f} |\n")
    viv = blob["grouped"]["VIVIDNESS"]; val = blob["grouped"]["VALUE"]; ansg = blob["grouped"]["ANSWERABILITY"]
    md(f"\n- VALUE {val:+.5f} | VIVIDNESS {viv:+.5f} | ANSWERABILITY {ansg:+.5f} | NOISE FLOOR {nf:+.5f}.\n"
       f"- **(ii) learner USES vividness:** {viv > nf + nfsd} (vividness group importance "
       f"{'>' if viv > nf + nfsd else '<='} noise floor +1sd).\n\n")

    # ---- margin fit (population val) ----
    if margin is None:
        print("[eval] fitting anchored margin on population val ...", flush=True)
        margin, mscores = fit_margin(env, tables, gbm, seed=0)
    else:
        mscores = {}
    md(f"## Anchored swap margin (fit on POPULATION val, firewall)\n\nFitted margin = **{margin}** "
       f"(grid {MARGIN_GRID}); pop-val scores {json.dumps({str(k): round(v,4) for k,v in mscores.items()})}.\n\n")

    # ---- four split-estimates ----
    est = {}
    pooled = {b: {arm: {"r": [], "s": []} for arm in ("free_vs_item", "anch_vs_item",
                                                       "free_vs_mixed", "anch_vs_mixed",
                                                       "free_vs_anch")} for b in BUDGETS}
    md("## Per split-estimate (anytime NDCG@10)\n\n")
    for seed in (0, 1):
        rng = np.random.default_rng(seed); perm = rng.permutation(n); half = n // 2
        A_idx = sorted(perm[:half].tolist()); B_idx = sorted(perm[half:].tolist())
        halves = {"A": A_idx, "B": B_idx}
        for evalhalf in ("A", "B"):
            other = "B" if evalhalf == "A" else "A"
            ev_users = [users[i] for i in halves[evalhalf]]
            ot_users = [users[i] for i in halves[other]]
            ev_cold = SC.cold_of(env, ev_users); ot_cold = SC.cold_of(env, ot_users)
            # SPLIT-CONSTRUCTED fair statics (built on OTHER half)
            s_item = SC.build_greedy_sub(env, item_pool, ot_users, ot_cold, TMAX)
            s_mixed = SC.build_greedy_sub(env, mixed_pool, ot_users, ot_cold, TMAX)
            pt_item = SC.eval_static_peruser(env, s_item, ev_users, ev_cold, TMAX)
            pt_mixed = SC.eval_static_peruser(env, s_mixed, ev_users, ev_cold, TMAX)
            # routers (anchored on s_item = the primary anchor family)
            pt_free = eval_router(env, tables, gbm, ev_users, ev_cold, TMAX, mode="free",
                                  answer_fn=afn)
            pt_anch = eval_router(env, tables, gbm, ev_users, ev_cold, TMAX, mode="anchored",
                                  s_fair=s_item, margin=margin, answer_fn=afn)
            key = f"seed{seed}_eval{evalhalf}"
            est[key] = {}
            for b in BUDGETS:
                ai = list(pt_item[:, :b].mean(axis=1)); am = list(pt_mixed[:, :b].mean(axis=1))
                af = list(pt_free[:, :b].mean(axis=1)); aa = list(pt_anch[:, :b].mean(axis=1))
                est[key][b] = dict(s_item=float(np.mean(ai)), s_mixed=float(np.mean(am)),
                                   r_free=float(np.mean(af)), r_anch=float(np.mean(aa)),
                                   free_vs_item=paired(af, ai), anch_vs_item=paired(aa, ai),
                                   free_vs_mixed=paired(af, am), anch_vs_mixed=paired(aa, am),
                                   free_vs_anch=paired(af, aa))
                pooled[b]["free_vs_item"]["r"] += af; pooled[b]["free_vs_item"]["s"] += ai
                pooled[b]["anch_vs_item"]["r"] += aa; pooled[b]["anch_vs_item"]["s"] += ai
                pooled[b]["free_vs_mixed"]["r"] += af; pooled[b]["free_vs_mixed"]["s"] += am
                pooled[b]["anch_vs_mixed"]["r"] += aa; pooled[b]["anch_vs_mixed"]["s"] += am
                pooled[b]["free_vs_anch"]["r"] += af; pooled[b]["free_vs_anch"]["s"] += aa
            e12 = est[key][PRIMARY_BUDGET]
            print(f"  [{key}] @12 s-item {e12['s_item']:.4f} s-mixed {e12['s_mixed']:.4f} "
                  f"r-free {e12['r_free']:.4f} r-anch {e12['r_anch']:.4f} | "
                  f"anch-vs-item {e12['anch_vs_item']['delta']:+.4f}"
                  f"[{e12['anch_vs_item']['ci'][0]:+.4f},{e12['anch_vs_item']['ci'][1]:+.4f}]", flush=True)
            md(f"### {key}\n\n| budget | s-item | s-mixed | r-free | r-anch | anch-vs-item [CI] | free-vs-item [CI] |\n"
               f"|---|--:|--:|--:|--:|---|---|\n")
            for b in BUDGETS:
                e = est[key][b]
                md(f"| T={b} | {e['s_item']:.4f} | {e['s_mixed']:.4f} | {e['r_free']:.4f} | {e['r_anch']:.4f} | "
                   f"{e['anch_vs_item']['delta']:+.4f}[{e['anch_vs_item']['ci'][0]:+.4f},{e['anch_vs_item']['ci'][1]:+.4f}] | "
                   f"{e['free_vs_item']['delta']:+.4f}[{e['free_vs_item']['ci'][0]:+.4f},{e['free_vs_item']['ci'][1]:+.4f}] |\n")
            md("\n")
            # incremental save
            json.dump(dict(estimates=est, margin=margin, importances=blob["grouped"],
                           noise_floor=blob["noise_floor"]), open(RESULT_JSON, "w"), indent=1, default=str)

    # ---- pooled verdicts ----
    pooled_ci = {b: {} for b in BUDGETS}
    for b in BUDGETS:
        for arm in pooled[b]:
            pooled_ci[b][arm] = paired(pooled[b][arm]["r"], pooled[b][arm]["s"])
    md("## POOLED verdicts (all 4 split-estimates concatenated)\n\n"
       "| contrast | T=8 | T=12 | T=24 |\n|---|---|---|---|\n")
    def cell(cb):
        return f"{cb['delta']:+.4f}[{cb['ci'][0]:+.4f},{cb['ci'][1]:+.4f}]"
    for arm, label in (("anch_vs_item", "r-anchored - s-item (FAIR)"),
                       ("free_vs_item", "r-free - s-item (FAIR)"),
                       ("anch_vs_mixed", "r-anchored - s-mixed (FAIR)"),
                       ("free_vs_mixed", "r-free - s-mixed (FAIR)"),
                       ("free_vs_anch", "r-free - r-anchored (expressiveness)")):
        md(f"| {label} | {cell(pooled_ci[8][arm])} | {cell(pooled_ci[12][arm])} | {cell(pooled_ci[24][arm])} |\n")
    md("\n")

    prim = pooled_ci[PRIMARY_BUDGET]
    a_vs_i = prim["anch_vs_item"]; f_vs_i = prim["free_vs_item"]; f_vs_a = prim["free_vs_anch"]
    win_i = bool(a_vs_i["ci"][0] > 0)
    e2_ok = bool(a_vs_i["ci"][0] >= -0.01 and f_vs_i["ci"][0] >= -0.01)
    expr_ok = bool(f_vs_a["ci"][0] >= -0.01)
    viv = blob["grouped"]["VIVIDNESS"]; nf = blob["noise_floor"]; nfsd = blob["feat_imp_sd"]["noise_floor"]
    uses_viv = bool(viv > nf + nfsd)
    md("## VERDICTS\n\n"
       f"- **(i) first fair learned-adaptivity win:** r-anchored - s-item @12 = {cell(a_vs_i)} -> "
       f"**{'WIN' if win_i else ('loses beyond noise' if a_vs_i['ci'][1] < -0.01 else 'TIE')}**.\n"
       f"- **(ii) learner uses vividness:** VIVIDNESS importance {viv:+.5f} vs noise floor {nf:+.5f} "
       f"(+1sd {nf+nfsd:+.5f}) -> **{uses_viv}**. (VALUE {blob['grouped']['VALUE']:+.5f}; "
       f"ANSWERABILITY {blob['grouped']['ANSWERABILITY']:+.5f}.)\n"
       f"- **(iii) expressiveness:** r-free - r-anchored @12 = {cell(f_vs_a)} -> free "
       f"{'does not lose beyond noise' if expr_ok else 'LOSES to anchored'}.\n"
       f"- **(iv) E2 floor:** neither router loses beyond noise -> **{e2_ok}**.\n\n")

    out = dict(banner="DIRECTIONAL 173/300", margin=margin, margin_scores=mscores,
               importances=blob["feat_imp"], grouped=blob["grouped"], noise_floor=blob["noise_floor"],
               va_r2=blob["va_r2"], n_examples=blob["n_examples"],
               pooled={str(b): {a: pooled_ci[b][a] for a in pooled_ci[b]} for b in BUDGETS},
               estimates=est, verdicts=dict(win_vs_item=win_i, uses_vividness=uses_viv,
                                            expressiveness_ok=expr_ok, e2_floor_ok=e2_ok),
               wall_min=round((time.time() - t0) / 60, 2))
    json.dump(out, open(RESULT_JSON, "w"), indent=1, default=str)
    print("\n==== POOLED @12 ====", flush=True)
    print(f"  r-anchored - s-item {cell(a_vs_i)} -> win={win_i}", flush=True)
    print(f"  r-free - s-item     {cell(f_vs_i)}", flush=True)
    print(f"  uses vividness = {uses_viv}; E2 floor = {e2_ok}; expressiveness = {expr_ok}", flush=True)
    print(f"\n[done] {OUT_MD} + {RESULT_JSON} (wall {out['wall_min']}m)", flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--do", choices=["train", "eval", "all"], default="all")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--margin", type=float, default=None)
    a = ap.parse_args()
    if a.do in ("train", "all"):
        train(a.seed)
    if a.do in ("eval", "all"):
        evaluate(a.margin)
