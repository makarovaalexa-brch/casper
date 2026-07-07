"""i25_phase4.py -- the DEFINITIVE adaptivity rerun on the VALID learned-fold instrument (I2.5 Phase 4).

Reuses the E0/E0d harness structure but replaces the additive belief operator z'=z+eta*a*q with the
LEARNED FOLD: at each turn the belief is z_t = fold(all answered tokens so far). Concepts fold through
the learned encoder (data-side aggregate tokens); the item channel = the user's KNOWN-RATED items answered
with their REAL centered ratings (the only honestly-answerable item answers, per E0f). NO LLM calls.

Arms: static B (rebuilt greedily under the new fold), static+skip, descent k=3/4/5 (realizable
|centered-rating| ranking + p_hat surrogate + a labelled per-user target-peek UPPER BOUND), emergent
switch (marginal-info x answerability x div), all-item-8. T=8, refusals cost a turn, NDCG@10 (+@50),
n~298 users, paired per-user bootstrap CIs.

Pre-registered verdict: does coarse->fine descent beat the best static under a valid learned fold. FINAL.

Run:  python scripts/i25_phase4.py
"""
import os, sys, json, time, collections
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import i25_lib as L
import i25_gates as GT
import answerability_main_study as MS

CKPT_BEST = ".cache/i25_fold_best.pt"
OUT_JSON = "experiments/I25_phase4.json"
T = 8
BOOT = 5000
SEED = 0


# ------------------------------------------------------------------ batched fold + NDCG
def fold_batch(FR, model, tok_lists, native_lists):
    if not tok_lists:
        return np.zeros((0, FR.W.shape[1]))
    tt, tv, te, mask, nz = L.pack_batch(FR, tok_lists, native_lists)
    with torch.no_grad():
        z = model(tt, tv, te, mask, nz)
    return z.numpy().astype(np.float64)


def ndcg_at(FR, z, tlike, profset, kk=10):
    S = FR.decode_np(z[None, :])[0].copy()
    S[list(profset)] = -1e9
    rel = set(tlike)
    if not rel:
        return None
    top = np.argpartition(-S, kk)[:kk]; top = top[np.argsort(-S[top])]
    W = 1.0 / np.log2(np.arange(2, kk + 2))
    dcg = sum(W[p] for p, t in enumerate(top) if int(t) in rel)
    idcg = W[:min(kk, len(rel))].sum() + 1e-12
    return dcg / idcg


# ------------------------------------------------------------------ per-user assembly
def assemble(D, FR):
    grid = json.load(open(L.GATE_GRID))["users"]
    split = L.G.build_split(D)
    concept_pct = MS.concept_pop_pct(D)
    pm = json.load(open(MS.PMODEL_JSON))
    pm_mean = np.array(pm["mean"]); pm_std = np.array(pm["std"]); pm_coef = np.array(pm["coef"]); pm_b = pm["intercept"]

    def phat(feats):
        z = (np.array(feats) - pm_mean) / pm_std
        return 1.0 / (1.0 + np.exp(-(z @ pm_coef + pm_b)))

    users = []
    concept_keys = set()
    for us, g in grid.items():
        u = int(us)
        if u not in split:
            continue
        kn, ho = split[u]; rat = dict(D["rat_by_u"][u])
        known = {j: rat[j] for j in kn if j in rat}
        like = [j for j in kn if rat.get(j, 0) >= 4]
        tlike = set(j for j in ho if rat.get(j, 0) >= 4)
        if not like or not tlike or len(known) < 4:
            continue
        cm = GT._cmean(known); ans = {int(k): v for k, v in g["ans"].items()}
        _, dgv = L.G.dominant_genre(D, sorted(kn), rat); nv = np.linalg.norm(dgv)
        # concept tokens (data-side aggregate over known members) + answerability + p_hat
        its = np.array(list(known.keys()))
        rel_all = D["concepts"]["item_tag"][its]
        crv = np.array([known[j] - cm for j in known])
        conc = {}
        for i, (k, m) in enumerate(g["Q"]):
            if k != "concept":
                continue
            ctag = m["ctag"]; key = f"C:{ctag}"; concept_keys.add(key)
            mass = float(rel_all[:, ctag].sum())
            val = float((rel_all[:, ctag] * crv).sum() / mass) if mass > 1e-6 else 0.0
            emb = FR.concept_emb(int(ctag))
            cv = L.G.concept_genre_vec(D, ctag); gm = float(dgv @ cv / nv) if nv > 0 else 0.0
            feats = [float(concept_pct[ctag]), float(np.log(D["concepts"]["coverage"][ctag] + 1.0)),
                     0.0, gm, 0.0, 1.0]
            conc[key] = dict(emb=emb, val=val, ans=bool(L.G.is_yes(ans.get(i, {}))), phat=float(phat(feats)))
        # item tokens: the user's known-rated items (answerable + real rating) + p_hat surrogate
        item_toks = []
        for j in known:
            gv = D["Gmat"][j].astype(np.float64); n2 = np.linalg.norm(gv)
            gm = float(dgv @ gv / (nv * n2)) if (nv > 0 and n2 > 0) else 0.0
            yr = MS.item_year(D["title"][j]); dec = ((yr - 1900) / 100.0) if yr else 0.5
            featsI = [float(D["pr"][j]), float(np.log(D["cnt"][j] + 1.0)), dec, gm,
                      float(MS.is_franchise(D["title"][j])), 0.0]
            item_toks.append(dict(j=j, emb=FR.Wn[j].numpy().astype(np.float32),
                                  val=float(known[j] - cm), like=known[j] >= 4,
                                  absr=abs(known[j] - cm), phat=float(phat(featsI))))
        users.append(dict(u=u, known=known, like=like, tlike=tlike, prof=set(kn),
                          conc=conc, items=item_toks))
    return users, sorted(concept_keys)


# token materializers ------------------------------------------------
def ctok(rec, key):
    c = rec["conc"][key]; return (1, c["emb"], c["val"], key)


def itok(it):
    return (0, it["emb"], it["val"], f"I:{it['j']}")


def eval_arm_curve(FR, model, users, pick_fn, kk=10):
    """pick_fn(rec) -> ordered list of (token, native_item_or_None) for up to T turns (already answerable).
    Returns per-user anytime (mean over t of NDCG) and endpoint arrays."""
    # build cumulative token sets per turn, batch-fold per turn across users
    plans = [pick_fn(rec) for rec in users]
    per_turn = np.full((len(users), T), np.nan)
    for t in range(T):
        tok_lists, nat_lists, idx = [], [], []
        for i, rec in enumerate(users):
            plan = plans[i][:t + 1]
            toks = [p[0] for p in plan]
            nat = [p[1] for p in plan if p[1] is not None]
            if not toks:
                continue
            tok_lists.append(toks); nat_lists.append(nat); idx.append(i)
        if not tok_lists:
            continue
        Z = fold_batch(FR, model, tok_lists, nat_lists)
        for r, i in enumerate(idx):
            per_turn[i, t] = ndcg_at(FR, Z[r], users[i]["tlike"], users[i]["prof"], kk)
    anytime = np.nanmean(per_turn, axis=1)
    endpoint = per_turn[:, -1]
    return anytime, endpoint, per_turn


def boot(a, b, seed=SEED):
    d = a - b; d = d[~np.isnan(d)]
    rng = np.random.default_rng(seed)
    bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(BOOT)])
    return dict(delta=float(d.mean()), ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                p_gt0=float((bs > 0).mean()), n=int(len(d)))


# ------------------------------------------------------------------ static schedule (greedy under fold)
def build_static(FR, model, users, concept_keys):
    """Greedy-forward shared concept schedule maximizing cohort-mean endpoint NDCG under the fold."""
    schedule = []
    def cohort_mean(sched):
        tok_lists, nat_lists, idx = [], [], []
        for i, rec in enumerate(users):
            toks = [ctok(rec, k) for k in sched if k in rec["conc"] and rec["conc"][k]["ans"]]
            if not toks:
                continue
            tok_lists.append(toks); nat_lists.append([]); idx.append(i)
        if not tok_lists:
            return 0.0
        Z = fold_batch(FR, model, tok_lists, nat_lists)
        vals = [ndcg_at(FR, Z[r], users[i]["tlike"], users[i]["prof"]) for r, i in enumerate(idx)]
        vals = [v for v in vals if v is not None]
        return float(np.mean(vals)) if vals else 0.0
    pool = list(concept_keys)
    for _ in range(T):
        best, bk = -1.0, None
        for k in pool:
            if k in schedule:
                continue
            v = cohort_mean(schedule + [k])
            if v > best:
                best, bk = v, k
        if bk is None:
            break
        schedule.append(bk)
    # tail priority order for static+skip (turn refund): remaining keys by single-key cohort value
    rest = [k for k in pool if k not in schedule]
    rest_val = {k: cohort_mean([k]) for k in rest}
    tail = sorted(rest, key=lambda k: -rest_val[k])
    return schedule, tail


# ------------------------------------------------------------------ static ITEM schedule (population-informed)
def build_item_static(FR, model, users, pool_size=60):
    """Strongest STATIC item schedule: one fixed item list for everyone, greedy forward on cohort-mean
    endpoint NDCG under the fold. A scheduled item folds for a user IFF it is in their KNOWN rated set
    (real centered rating); otherwise refusal (turn consumed). Candidate pool = top items by cohort
    coverage (population answerability: how many study users rated it in the known half)."""
    cover = collections.Counter()
    tok_of = {}                                   # (u_idx, j) -> item token dict
    for i, rec in enumerate(users):
        for it in rec["items"]:
            cover[it["j"]] += 1
            tok_of[(i, it["j"])] = it
    pool = [j for j, _ in cover.most_common(pool_size)]

    def cohort_mean_items(sched):
        tok_lists, nat_lists, idx = [], [], []
        for i, rec in enumerate(users):
            toks, nat = [], []
            for j in sched:
                it = tok_of.get((i, j))
                if it is not None:
                    toks.append(itok(it))
                    if it["like"]:
                        nat.append(j)
            if not toks:
                continue
            tok_lists.append(toks); nat_lists.append(nat); idx.append(i)
        if not tok_lists:
            return 0.0
        Z = fold_batch(FR, model, tok_lists, nat_lists)
        vals = [ndcg_at(FR, Z[r], users[i]["tlike"], users[i]["prof"]) for r, i in enumerate(idx)]
        vals = [v for v in vals if v is not None]
        return float(np.mean(vals)) if vals else 0.0

    schedule = []
    for _ in range(T):
        best, bj = -1.0, None
        for j in pool:
            if j in schedule:
                continue
            v = cohort_mean_items(schedule + [j])
            if v > best:
                best, bj = v, j
        if bj is None:
            break
        schedule.append(bj)
    cov_pct = [round(cover[j] / len(users), 3) for j in schedule]
    return schedule, cov_pct


def pick_item_static(item_schedule):
    def f(rec):
        known_j = {it["j"]: it for it in rec["items"]}
        plan = []
        for j in item_schedule:
            it = known_j.get(j)
            if it is not None:
                plan.append((itok(it), j if it["like"] else None))
            else:
                plan.append(("REFUSE", None))       # unrated by this user -> refusal, turn consumed
        return plan
    return f


# ------------------------------------------------------------------ pick functions (arms)
def pick_static(schedule, skip=False, tail=()):
    """skip=False: frozen schedule; a refusal consumes the turn (REFUSE slot).
    skip=True: conditional static — walk schedule++tail priority list, ask only answerable entries
    (the true-table realization of 'a refused turn is refunded to the next item on the list')."""
    def f(rec):
        plan = []
        if skip:
            for k in list(schedule) + list(tail):
                if len(plan) >= T:
                    break
                if k in rec["conc"] and rec["conc"][k]["ans"]:
                    plan.append((ctok(rec, k), None))
            return plan
        for k in schedule:
            if k in rec["conc"] and rec["conc"][k]["ans"]:
                plan.append((ctok(rec, k), None))
            else:
                plan.append(("REFUSE", None))        # refusal consumes the turn
        return plan
    return f


def pick_descent(schedule, k_switch, item_rank="absr", peek=False):
    def f(rec):
        plan = []
        # phase 1: k_switch concepts from the static schedule (answerable, else refusal)
        cnt = 0
        for key in schedule:
            if cnt >= k_switch:
                break
            if key in rec["conc"] and rec["conc"][key]["ans"]:
                plan.append((ctok(rec, key), None))
            else:
                plan.append(("REFUSE", None))
            cnt += 1
        # phase 2: known-rated items, ranked
        items = rec["items"][:]
        if peek:
            items = rank_by_peek(rec, items)
        elif item_rank == "absr":
            items = sorted(items, key=lambda it: -it["absr"])
        elif item_rank == "phat":
            items = sorted(items, key=lambda it: -(it["phat"] * it["absr"]))
        need = T - len(plan)
        for it in items[:max(need, 0)]:
            plan.append((itok(it), it["j"] if it["like"] else None))
        return plan
    return f


_PEEK_CACHE = {}
def rank_by_peek(rec, items):
    """UPPER BOUND (leaky): rank the user's known items by their own single-item held-out NDCG gain."""
    key = rec["u"]
    if key in _PEEK_CACHE:
        order = _PEEK_CACHE[key]
        return [it for _, it in sorted(zip(order, items), key=lambda x: -x[0])]
    return items  # filled in main (needs FR/model); see compute_peek


def pick_all_item(item_rank="absr"):
    def f(rec):
        items = sorted(rec["items"], key=lambda it: -it["absr"]) if item_rank == "absr" else rec["items"]
        return [(itok(it), it["j"] if it["like"] else None) for it in items[:T]]
    return f


def pick_emergent(FR, model, concept_keys):
    """Greedy: score = marginal_info(candidate | answered span) x answerability(true-table gate) x div.
    div(q) = q^T Cov(W) q over decoder rows (E0/E0d divisiveness), clamped at 0."""
    Wm = FR.W.numpy().astype(np.float64)
    Wc = Wm - Wm.mean(0, keepdims=True)
    C = (Wc.T @ Wc) / Wm.shape[0]
    def f(rec):
        answered_dirs = []
        plan = []
        # candidate pool: answerable concepts (emb) + known items (Wn); div precomputed (turn-invariant)
        cands = []
        for k, c in rec["conc"].items():
            if c["ans"]:
                e = np.asarray(c["emb"], float); e = e / (np.linalg.norm(e) + 1e-9)
                cands.append(("C", k, e, max(float(e @ C @ e), 0.0)))
        for it in rec["items"]:
            e = np.asarray(it["emb"], float); e = e / (np.linalg.norm(e) + 1e-9)
            cands.append(("I", it["j"], e, max(float(e @ C @ e), 0.0)))
        used = set()
        for _ in range(T):
            best, bpick = -1e9, None
            span = np.array(answered_dirs) if answered_dirs else None
            for typ, kid, e, dv in cands:
                tag = (typ, kid)
                if tag in used:
                    continue
                if span is not None and len(span):
                    resid = e - span.T @ (span @ e)
                    mi = float(np.linalg.norm(resid))
                else:
                    mi = 1.0
                score = mi * dv
                if score > best:
                    best, bpick = score, (typ, kid, e)
            if bpick is None:
                break
            typ, kid, e = bpick; used.add((typ, kid))
            answered_dirs.append(e)
            # orthonormalize span
            span2 = []
            for v in answered_dirs:
                w = v.copy()
                for u2 in span2:
                    w = w - (w @ u2) * u2
                nrm = np.linalg.norm(w)
                if nrm > 1e-6:
                    span2.append(w / nrm)
            answered_dirs = span2 if span2 else answered_dirs
            if typ == "C":
                plan.append((ctok(rec, kid), None))
            else:
                it = next(x for x in rec["items"] if x["j"] == kid)
                plan.append((itok(it), it["j"] if it["like"] else None))
        return plan
    return f


# refusal token handling: a "REFUSE" placeholder folds nothing (belief unchanged) but consumes the turn
def clean_plan(plan):
    """Turn REFUSE placeholders into 'no new token this turn' by carrying the prior token set.
    We implement by dropping the token but keeping the turn slot: eval_arm_curve slices plan[:t+1],
    so a REFUSE contributes no token yet occupies a slot -> belief unchanged that turn."""
    out = []
    for tok, nat in plan:
        if tok == "REFUSE" or tok is None:
            out.append((None, None))
        else:
            out.append((tok, nat))
    return out


def wrap(pick_fn):
    def f(rec):
        return clean_plan(pick_fn(rec))
    return f


# eval_arm_curve must tolerate None tokens (refusal slots) --------------------
def eval_arm(FR, model, users, pick_fn, kk=10):
    plans = [clean_plan(pick_fn(rec)) for rec in users]
    answered = np.array([sum(1 for p in pl[:T] if p[0] is not None) for pl in plans], float)
    per_turn = np.full((len(users), T), np.nan)
    for t in range(T):
        tok_lists, nat_lists, idx = [], [], []
        for i, rec in enumerate(users):
            plan = plans[i][:t + 1]
            toks = [p[0] for p in plan if p[0] is not None]
            nat = [p[1] for p in plan if p[0] is not None and p[1] is not None]
            if not toks:
                continue
            tok_lists.append(toks); nat_lists.append(nat); idx.append(i)
        if not tok_lists:
            continue
        Z = fold_batch(FR, model, tok_lists, nat_lists)
        for r, i in enumerate(idx):
            per_turn[i, t] = ndcg_at(FR, Z[r], users[i]["tlike"], users[i]["prof"], kk)
    return np.nanmean(per_turn, axis=1), per_turn[:, -1], per_turn


def compute_peek(FR, model, users):
    """Fill _PEEK_CACHE: per user, single-item held-out NDCG gain for each known item."""
    for rec in users:
        gains = []
        base = ndcg_at(FR, np.zeros(FR.W.shape[1]), rec["tlike"], rec["prof"]) or 0.0
        tok_lists = [[itok(it)] for it in rec["items"]]
        nat_lists = [[it["j"]] if it["like"] else [] for it in rec["items"]]
        Z = fold_batch(FR, model, tok_lists, nat_lists)
        for r, it in enumerate(rec["items"]):
            v = ndcg_at(FR, Z[r], rec["tlike"], rec["prof"]) or 0.0
            gains.append(v - base)
        _PEEK_CACHE[rec["u"]] = gains


def rank_by_peek(rec, items):   # override the stub
    gains = _PEEK_CACHE.get(rec["u"])
    if gains is None:
        return items
    return [it for _, it in sorted(zip(gains, items), key=lambda x: -x[0])]


def main():
    t0 = time.time()
    print("[p4] loading data + frozen RecVAE + trained fold ...", flush=True)
    D = L.G.load_data(); FR = L.Frozen(D)
    model = L.Fold(); blob = torch.load(CKPT_BEST, map_location="cpu")
    model.load_state_dict(blob["model"]); model.eval()
    users, concept_keys = assemble(D, FR)
    print(f"[p4] {len(users)} users, {len(concept_keys)} shared concept keys, "
          f"fold best val {blob['state']['best_val']:.4f}", flush=True)

    print("[p4] building static schedule greedily under the fold ...", flush=True)
    schedule, tail = build_static(FR, model, users, concept_keys)
    print(f"   static schedule: {schedule}", flush=True)
    print(f"   skip tail head: {tail[:8]}", flush=True)
    compute_peek(FR, model, users)

    arms = {}
    arms["static B"] = eval_arm(FR, model, users, pick_static(schedule, skip=False))
    arms["static+skip"] = eval_arm(FR, model, users, pick_static(schedule, skip=True, tail=tail))
    for k in (3, 4, 5):
        arms[f"descent k={k} (|rating|)"] = eval_arm(FR, model, users, pick_descent(schedule, k, "absr"))
    arms["descent k=4 (p-hat)"] = eval_arm(FR, model, users, pick_descent(schedule, 4, "phat"))
    arms["descent k=4 (peek UB, leaky)"] = eval_arm(FR, model, users, pick_descent(schedule, 4, peek=True))
    arms["emergent (marginal-info)"] = eval_arm(FR, model, users, pick_emergent(FR, model, concept_keys))
    arms["all-item-8 (|rating|)"] = eval_arm(FR, model, users, pick_all_item())

    ref = arms["static B"]
    ref_skip = arms["static+skip"]
    best_static_any = np.nanmean(np.where(np.isnan(ref[0]), ref_skip[0], np.maximum(ref[0], ref_skip[0])))
    rows = {}
    for name, (any_a, end_a, _) in arms.items():
        vs_B = boot(any_a, ref[0])
        vs_skip = boot(any_a, ref_skip[0])
        rows[name] = dict(anytime=float(np.nanmean(any_a)), endpoint=float(np.nanmean(end_a)),
                          vs_staticB=vs_B, vs_staticskip=vs_skip)
    # NDCG@50 for the headline arms (cheap)
    at50 = {}
    for name in ("static B", "descent k=4 (|rating|)", "all-item-8 (|rating|)"):
        pf = {"static B": pick_static(schedule, False),
              "descent k=4 (|rating|)": pick_descent(schedule, 4, "absr"),
              "all-item-8 (|rating|)": pick_all_item()}[name]
        a, e, _ = eval_arm(FR, model, users, pf, kk=50)
        at50[name] = dict(anytime=float(np.nanmean(a)), endpoint=float(np.nanmean(e)))

    # pre-registered verdict: best realizable descent vs best static (both anytime)
    desc_names = [n for n in rows if n.startswith("descent") and "peek" not in n]
    best_desc = max(desc_names, key=lambda n: rows[n]["anytime"])
    best_static_name = "static B" if rows["static B"]["anytime"] >= rows["static+skip"]["anytime"] else "static+skip"
    verdict = boot(arms[best_desc][0], arms[best_static_name][0])
    flips = verdict["ci"][0] > 0 and verdict["delta"] >= 0.010

    out = dict(config=dict(dataset="ML-25M", instrument="RecVAE-d512 + I2.5 learned fold", T=T,
                           n_users=len(users), belief="z_t = fold(answered tokens so far)",
                           item_channel="user's KNOWN-RATED items, real centered rating (E0f realization)",
                           fold_ckpt=CKPT_BEST, best_val=blob["state"]["best_val"]),
               static_schedule=schedule, arms=rows, ndcg_at50=at50,
               verdict=dict(best_descent=best_desc, best_static=best_static_name, delta_anytime=verdict,
                            descent_beats_static=bool(flips),
                            rule="descent wins iff anytime Delta>=0.010 AND CI excludes 0"),
               wall_min=round((time.time() - t0) / 60, 2))
    os.makedirs("experiments", exist_ok=True)
    json.dump(out, open(OUT_JSON, "w"), indent=1, default=str)

    print("\n==== PHASE 4 ARMS (anytime / endpoint NDCG@10) ====", flush=True)
    for name, r in rows.items():
        print(f"  {name:32s} any {r['anytime']:.4f} end {r['endpoint']:.4f} "
              f"| vs B {r['vs_staticB']['delta']:+.4f} CI[{r['vs_staticB']['ci'][0]:+.4f},"
              f"{r['vs_staticB']['ci'][1]:+.4f}]", flush=True)
    print(f"\n  VERDICT: best descent [{best_desc}] vs best static [{best_static_name}]: "
          f"anytime d={verdict['delta']:+.4f} CI[{verdict['ci'][0]:+.4f},{verdict['ci'][1]:+.4f}] "
          f"-> descent_beats_static={flips}", flush=True)
    print(f"  wall {out['wall_min']}m -> {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
