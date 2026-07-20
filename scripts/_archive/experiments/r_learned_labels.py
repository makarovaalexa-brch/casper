"""r_learned_labels.py -- LABEL GENERATION + POPULATION FEATURE TABLES for r-learned.

THE LEARNED BELIEF-COMBINER (author-specified): value-belief and vividness-belief enter as FEATURES;
the combination is LEARNED from NDCG outcome labels; NOTHING optimizes for answerability per se.

FIREWALL (E5): everything here is computed on the trU POPULATION (the meta 'trU' split, ~24.8k
users). The 300 study users (va/te) are NEVER touched. The LLM-judged grid is NEVER an input. No
held-out study target ever enters a feature. The candidate UNIVERSE is the SAME 180 CANDS entities
used at eval (items / genome-concepts / attributes) so every feature transfers by construction, but
the ANSWERS are simulated from population ratings via the certified fold-v2 sampler (build_reveal_v2
distribution: real ratings -> data tokens; member-aggregates -> llm tokens; sigma=0.70 noise).

WHAT A DEPLOYED SYSTEM COMPUTES (documented; all population, all blind, none study-derived):
  - pop_value[cd]        = population mean single-answer NDCG@10 lift from cold (per entity).
  - pop_answerability[cd]= population P(answer, k>=1) = fraction of population users who answer cd.
  - pop_vividness[cd]    = population P(k>=2) = pop_answerability[cd] * base_kw[channel]
                           (base_kw = the certified CALIB: item .700 / concept .521 / attr .363).
  - taste                = cos(z, q_dir), q_dir = candidate embedding (state-dependent, blind).
  - taste x {turn, answerability, vividness}  (the genre/taste-match interactions).
None uses study data or held-out targets. The learner is free to weight value vs vividness vs
answerability however the outcome labels dictate -- pre-registered read (ii) measures which it uses.

LABEL = 1-step realized NDCG@10 gain: fold(z + answer_to_q) - fold(z) against the SIMULATED user's
held-out likes. Refused candidate -> gain = 0 exactly (no-op turn). SELECTION-TIME features ONLY
(the realized answer value/fidelity is NOT a feature -- you must pick BEFORE the answer is revealed).

Chunks (~20k examples) persisted to .cache/r_learned_labels/chunk_*.npz so an interruption never
loses more than one chunk. Feature tables cached to .cache/r_learned_feat.npz.

NO LLM calls; all local.
Run:  python scripts/r_learned_labels.py --n_users 2800 --seed 0
"""
import os, sys, json, time, argparse, collections
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import warnings
warnings.filterwarnings("ignore")
import repair_probes as RP
import i25_fold_v2 as V2
import i25_lib as L
import i25_phase4_fair as FA
import battery_stage_c as SC

LAB_DIR = ".cache/r_learned_labels"
FEAT_CACHE = ".cache/r_learned_feat.npz"
CHUNK_EXAMPLES = 20000
TMAX = 12
SIGMA_STAR = V2.SIGMA_STAR
P_FAMOUS_INFER = 0.60          # a deployed answerer infers about an unrated FAMOUS item w.p. this

# ---- feature layout (IDENTICAL in label-gen and at eval) ------------------------------------------
FEAT_NAMES = [
    "ch_item", "ch_concept", "ch_attr",     # 0-2  channel one-hot
    "pop_value",                             # 3    population mean single-answer lift (per entity)
    "pop_answerability",                     # 4    population P(k>=1)
    "pop_vividness",                         # 5    population P(k>=2) = answerability * base_kw[ch]
    "taste",                                 # 6    cos(z, q_dir)
    "taste_x_turn",                          # 7    taste * (turn/12)
    "taste_x_answerability",                 # 8    taste-adjusted answerability
    "taste_x_vividness",                     # 9    taste/genre match x vividness
    "value_tier",                            # 10   pop_value percentile in [0,1]
    "turn",                                  # 11   turn index 0..11
    "norm_z",                                # 12   ||z||
    "answered_count",                        # 13   answers folded so far
    "noise_floor",                           # 14   random ~U(0,1) -- the importance NOISE FLOOR
]
NF = len(FEAT_NAMES)
FEAT_GROUPS = {
    "VALUE": ["pop_value", "value_tier"],
    "VIVIDNESS": ["pop_vividness", "taste_x_vividness"],
    "ANSWERABILITY": ["pop_answerability", "taste_x_answerability"],
    "TASTE": ["taste", "taste_x_turn"],
    "STATE": ["turn", "norm_z", "answered_count"],
    "CHANNEL": ["ch_item", "ch_concept", "ch_attr"],
    "NOISE_FLOOR": ["noise_floor"],
}
BASE_KW = V2.CALIB["p_kw_given_ans"]          # {'item':0.700,'concept':0.521,'attr':0.363}
CH_NAME = {0: "item", 1: "concept", 2: "attr"}


# ==================================================================== population environment
_POP = None
def pop_env():
    """Frozen encoder + v2 fold + the CANDS universe (from RP.setup = the EVAL universe) + population
    item-mean + famous set + per-cand member bags (to match the embeddings)."""
    global _POP
    if _POP is not None:
        return _POP
    env = RP.setup()
    D, FR, model, CANDS, memb, battery = (env["D"], env["FR"], env["model"], env["CANDS"],
                                          env["memb"], env["battery"])
    item_mean = V2.load_item_mean(D)
    ni = int(D["ni"])
    famous = set(int(j) for j in np.where(D["tier"] == "famous")[0])
    # per-cand member item bag (dense ids) -- item: {j}; concept: genome membership; attr: battery bag
    for m in CANDS:
        if m["typ"] == 0:
            m["members"] = np.array([int(m["key"])], np.int64)
            m["base_kw"] = BASE_KW["item"]
        elif m["typ"] == 1:
            mems = [j for j in memb.get(str(int(m["key"])), []) if 0 <= j < ni]
            m["members"] = np.array(mems, np.int64); m["base_kw"] = BASE_KW["concept"]
        else:
            e = battery.get(m["key"]) or {}
            mems = [j for j in e.get("member_dense_ids", []) if 0 <= j < ni]
            m["members"] = np.array(mems, np.int64); m["base_kw"] = BASE_KW["attr"]
    _POP = dict(env=env, D=D, FR=FR, model=model, CANDS=CANDS, item_mean=item_mean,
                famous=famous, ni=ni)
    print(f"[pop_env] {len(CANDS)} CANDS "
          f"({sum(m['typ']==0 for m in CANDS)}i/{sum(m['typ']==1 for m in CANDS)}c/"
          f"{sum(m['typ']==2 for m in CANDS)}a); ni={ni}", flush=True)
    return _POP


# ==================================================================== simulate a population user's
#                                                                      answers over the CANDS universe
def simulate_answers(P, known, rng):
    """Population user's known real ratings {j:stars} -> per-cand simulated answer, mirroring
    build_reveal_v2 per entity. Returns cd -> (token, native_j_or_None) for ANSWERED cands only.
      item rated       -> data token (real centered rating)
      item famous-unrated -> llm token w.p. P_FAMOUS_INFER (item_mean + N(0,sigma))
      concept/attr     -> member-aggregate llm token if the user rated >=2 of its member items."""
    D, FR, CANDS, item_mean, famous = P["D"], P["FR"], P["CANDS"], P["item_mean"], P["famous"]
    if len(known) < 2:
        return {}
    kk = np.fromiter(known.keys(), np.int64)
    mu = float(np.mean([known[j] for j in kk]))
    cr = {int(j): known[int(j)] - mu for j in kk}
    kset = set(int(j) for j in kk)
    out = {}
    for m in CANDS:
        cd = m["cid"]; typ = m["typ"]
        if typ == 0:
            j = int(m["key"])
            if j in kset:
                out[cd] = ((0, V2.FID["data"], m["emb"], float(cr[j])),
                           j if known[j] >= 4 else None)
            elif j in famous and rng.random() < P_FAMOUS_INFER:
                fid, _ = V2.sample_knowledge("item", rng)
                v = float(np.clip(item_mean[j] + rng.normal(0, SIGMA_STAR), 0.5, 5.0) - mu)
                out[cd] = ((0, fid, m["emb"], v), None)
        else:
            mems = m["members"]
            if len(mems) == 0:
                continue
            known_mem = [int(j) for j in mems if int(j) in kset]
            if len(known_mem) >= 2:
                a = float(np.mean([cr[j] for j in known_mem]))       # centered aggregate (stars-space)
                val = V2.bin_star(a + rng.normal(0, SIGMA_STAR) + mu)
                fid, _ = V2.sample_knowledge(CH_NAME[typ], rng)
                out[cd] = ((typ, fid, m["emb"], float(val)), None)
    return out


# ==================================================================== population feature tables
def build_feature_tables(P, n_users=4000, seed=0):
    if os.path.exists(FEAT_CACHE):
        z = np.load(FEAT_CACHE)
        print(f"[feat] loaded cached tables ({FEAT_CACHE})", flush=True)
        return {k: z[k] for k in ("pop_value", "pop_answerability", "pop_vividness", "value_tier")}
    D, FR, model, CANDS = P["D"], P["FR"], P["model"], P["CANDS"]
    nc = len(CANDS)
    prof = L.load_train_profiles(D, n_users, seed=seed + 900)
    rng = np.random.default_rng(seed + 901)
    users = []
    for u, p in prof.items():
        known, held = V2.make_user_split(p, rng)
        if held and len(known) >= 4:
            users.append(dict(known=known, held=held))
    print(f"[feat] {len(users)} population users for feature tables", flush=True)
    lift_sum = np.zeros(nc); ans_cnt = np.zeros(nc, np.int64)
    tl, nl, meta, recs_for, cold_for = [], [], [], [], []

    def flush():
        for s in range(0, len(tl), 3000):
            e = min(s + 3000, len(tl))
            Z = V2.fold_batch_v2(FR, model, tl[s:e], nl[s:e])
            vals = FA.ndcg_batch(FR, Z, recs_for[s:e], 10)
            for r in range(s, e):
                cd = meta[r]
                lift_sum[cd] += vals[r - s] - cold_for[r]
                ans_cnt[cd] += 1

    for rec in users:
        recu = dict(tlike=rec["held"], prof=set(rec["known"].keys()))
        cold = float(FA.ndcg_batch(FR, np.zeros((1, FR.W.shape[1]), np.float32), [recu], 10)[0])
        ans = simulate_answers(P, rec["known"], rng)
        for cd, (tok, nat) in ans.items():
            tl.append([tok]); nl.append([nat] if nat is not None else [])
            meta.append(cd); recs_for.append(recu); cold_for.append(cold)
        if len(tl) >= 4000:
            flush(); tl.clear(); nl.clear(); meta.clear(); recs_for.clear(); cold_for.clear()
    flush()
    n_pop = len(users)
    pop_value = np.where(ans_cnt > 0, lift_sum / np.maximum(ans_cnt, 1), 0.0)
    pop_answerability = ans_cnt.astype(np.float64) / max(n_pop, 1)
    base_kw = np.array([CANDS[c]["base_kw"] for c in range(nc)])
    pop_vividness = pop_answerability * base_kw
    value_tier = np.argsort(np.argsort(pop_value)) / max(nc - 1, 1)
    np.savez(FEAT_CACHE, pop_value=pop_value, pop_answerability=pop_answerability,
             pop_vividness=pop_vividness, value_tier=value_tier, ans_cnt=ans_cnt,
             n_pop=np.array([n_pop]))
    print(f"[feat] built + cached. pop_value [{pop_value.min():.4f},{pop_value.max():.4f}]; "
          f"answerability [{pop_answerability.min():.3f},{pop_answerability.max():.3f}]", flush=True)
    return dict(pop_value=pop_value, pop_answerability=pop_answerability,
                pop_vividness=pop_vividness, value_tier=value_tier)


# ==================================================================== the shared feature function
def make_features(P, tables, cd, z, t, answered_count, noise_rng=None):
    """SELECTION-TIME features for candidate cd at state (z, turn t). NO realized answer, NO study
    data. Identical in label-gen and at eval."""
    m = P["CANDS"][cd]; emb = m["emb"]; typ = m["typ"]
    nz = float(np.linalg.norm(z))
    taste = float(z @ emb / (nz * (np.linalg.norm(emb) + 1e-9))) if nz > 0 else 0.0
    ans = float(tables["pop_answerability"][cd]); viv = float(tables["pop_vividness"][cd])
    tgate = 0.5 * (taste + 1.0)               # cos in [-1,1] -> [0,1]
    f = np.zeros(NF, np.float64)
    f[0] = 1.0 if typ == 0 else 0.0
    f[1] = 1.0 if typ == 1 else 0.0
    f[2] = 1.0 if typ == 2 else 0.0
    f[3] = float(tables["pop_value"][cd])
    f[4] = ans
    f[5] = viv
    f[6] = taste
    f[7] = taste * (t / TMAX)
    f[8] = ans * tgate
    f[9] = viv * tgate
    f[10] = float(tables["value_tier"][cd])
    f[11] = float(t)
    f[12] = nz
    f[13] = float(answered_count)
    f[14] = (noise_rng.random() if noise_rng is not None else np.random.random())
    return f


# ==================================================================== label generation
def generate(n_users=2800, seed=0, cands_per_state=8, refused_per_state=3):
    P = pop_env()
    tables = build_feature_tables(P, n_users=4000, seed=seed)
    FR, model, CANDS = P["FR"], P["model"], P["CANDS"]
    os.makedirs(LAB_DIR, exist_ok=True)
    prof = L.load_train_profiles(P["D"], n_users, seed=seed)
    rng = np.random.default_rng(seed + 1)
    noise_rng = np.random.default_rng(seed + 2)
    users = []
    for u, p in prof.items():
        known, held = V2.make_user_split(p, rng)
        if held and len(known) >= 4:
            users.append(dict(u=u, known=known, held=held))
    print(f"[gen] {len(users)} population users; ~200k target; {cands_per_state} answered + "
          f"{refused_per_state} refused per state", flush=True)

    st = dict(chunk=0, total=0, t0=time.time())
    Xb, Yb, Tb, Cb, Ub = [], [], [], [], []

    def save_chunk():
        if not Xb:
            return
        path = os.path.join(LAB_DIR, f"chunk_{seed}_{st['chunk']:03d}.npz")
        np.savez(path, X=np.array(Xb, np.float32), Y=np.array(Yb, np.float32),
                 turn=np.array(Tb, np.int16), ch=np.array(Cb, np.int16), user=np.array(Ub, np.int64))
        print(f"[gen] wrote {path} ({len(Xb)} ex; total {st['total']}; "
              f"{(time.time()-st['t0'])/60:.1f}m)", flush=True)
        st["chunk"] += 1; Xb.clear(); Yb.clear(); Tb.clear(); Cb.clear(); Ub.clear()

    def emit(feat, gain, t, ch, uid):
        Xb.append(feat); Yb.append(gain); Tb.append(t); Cb.append(ch); Ub.append(uid); st["total"] += 1

    for rec in users:
        urng = np.random.default_rng(seed * 100003 + int(rec["u"]))
        ans = simulate_answers(P, rec["known"], urng)
        answered_cds = list(ans.keys())
        if len(answered_cds) < 2:
            continue
        recu = dict(tlike=rec["held"], prof=set(rec["known"].keys()))
        cold = float(FA.ndcg_batch(FR, np.zeros((1, FR.W.shape[1]), np.float32), [recu], 10)[0])
        refused_cds = [m["cid"] for m in CANDS if m["cid"] not in ans]
        ev_order = list(urng.permutation(answered_cds))
        state_tl, state_nl, state_meta, prior_z = [], [], [], {}
        for t in range(TMAX):
            ev = ev_order[:t]
            ev_toks = [ans[c][0] for c in ev]
            ev_nat = [ans[c][1] for c in ev if ans[c][1] is not None]
            z = V2.fold_np_v2(FR, model, ev_toks, ev_nat) if ev_toks else np.zeros(FR.W.shape[1])
            prior_z[t] = z
            base = float(FA.ndcg_batch(FR, z[None, :].astype(np.float32), [recu], 10)[0]) if ev_toks else cold
            av_ans = [c for c in answered_cds if c not in ev]
            for c in list(urng.permutation(av_ans))[:cands_per_state]:
                tok, nat = ans[c]
                state_tl.append(ev_toks + [tok])
                state_nl.append(ev_nat + ([nat] if nat is not None else []))
                state_meta.append((t, c, base))
            for c in (list(urng.permutation(refused_cds))[:refused_per_state] if refused_cds else []):
                emit(make_features(P, tables, c, z, t, len(ev), noise_rng), 0.0, t, CANDS[c]["typ"],
                     int(rec["u"]))
        for s in range(0, len(state_tl), 3000):
            e = min(s + 3000, len(state_tl))
            Z = V2.fold_batch_v2(FR, model, state_tl[s:e], state_nl[s:e])
            vals = FA.ndcg_batch(FR, Z, [recu] * (e - s), 10)
            for r in range(s, e):
                t, c, base = state_meta[r]
                emit(make_features(P, tables, c, prior_z[t], t, t, noise_rng),
                     float(vals[r - s] - base), t, CANDS[c]["typ"], int(rec["u"]))
        if len(Xb) >= CHUNK_EXAMPLES:
            save_chunk()
    save_chunk()
    manifest = dict(n_users=len(users), seed=seed, total=st["total"], n_chunks=st["chunk"],
                    feat_names=FEAT_NAMES, cands_per_state=cands_per_state,
                    refused_per_state=refused_per_state, wall_min=round((time.time()-st["t0"])/60, 2))
    json.dump(manifest, open(os.path.join(LAB_DIR, f"manifest_{seed}.json"), "w"), indent=1)
    print(f"[gen] DONE. {st['total']} examples in {st['chunk']} chunks -> {LAB_DIR} "
          f"({manifest['wall_min']}m)", flush=True)
    return manifest


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_users", type=int, default=2800)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cands_per_state", type=int, default=8)
    ap.add_argument("--refused_per_state", type=int, default=3)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--feat_only", action="store_true")
    a = ap.parse_args()
    if a.threads > 0:
        torch.set_num_threads(a.threads)
    if a.feat_only:
        build_feature_tables(pop_env(), n_users=4000, seed=a.seed)
    else:
        generate(a.n_users, a.seed, a.cands_per_state, a.refused_per_state)
