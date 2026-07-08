"""battery_stage_b.py -- STAGE B of the DIRECTIONAL adaptivity battery: DISCOVERY FEASIBILITY.

Retargets the KMAP_OFFLINE protocol to the NEW answerer-v1 judged grid (173/300 users, DIRECTIONAL).
Question: can a firewall-clean belief PREDICT held-out judged answerability per user, and how fast
(AUC-vs-t) + how high (split-half asymptote)? Compared to the OLD arena's rated-ness ceiling 0.725.

Three beliefs per channel (E5 circularity firewall -- NEVER the eval user's own cells for training):
  (i)   popularity/prominence baseline (population answer-rate logit; the t=0 predictor).
  (ii)  kmap POPULATION rated-matrix embeddings (kmap_emb.npz; online k_u infer) -- items; concept
        centroid e_c = member-item mean for concepts.
  (iii) NEW LEAVE-ONE-USER-OUT judged-grid MF: logistic MF d=16 on the OTHER users' judged matrix
        (5-fold over users so the eval user's cells never train the embedding), online k_u infer.

Targets: held-out judged answerability, k>=1 and k>=2, ITEMS and CONCEPTS separately.
Protocol: reveal t in {0,2,4,8,16} events in popularity-desc order; infer the belief's user vector
from those t events; predict HELD-OUT cells; per-user AUC (needs both classes); mean. Split-half
asymptote = infer on a random half of events, predict the other half. Paired per-user bootstrap.

NO LLM calls. Called via adaptivity_battery_v1.py --stage b.
"""
import os, sys, json, time, collections
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import roc_auc_score
import adaptivity_battery_v1 as A
import llm_answerability_gate as G

EMB = ".cache/instrument2/kmap_emb.npz"
INT = ".cache/instrument2/kmap_intercepts.npz"
TAG_MEMB = ".cache/instrument2/tag_membership.json"
OUT_JSON = "experiments/adaptivity_battery_v1_B.json"

TS = [0, 2, 4, 8, 16]
TAU = 1.0; TAU_A = 2.0; NEWTON_IT = 12
D_MF = 16; MF_EPOCHS = 60; MF_LR = 0.05; MF_L2 = 1e-4; N_NEG = 3
NFOLD = 5
BOOT = A.BOOT; SEED = A.SEED


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def infer_user(E_ev, b_ev, y_ev, d):
    """MAP-Newton for x=[alpha,k] given events with embeddings E_ev, intercepts b_ev, labels y_ev."""
    x = np.zeros(1 + d)
    if len(y_ev) == 0:
        return 0.0, np.zeros(d)
    Amat = np.column_stack([np.ones(len(y_ev)), E_ev])
    pv = np.concatenate([[TAU_A ** 2], np.full(d, TAU ** 2)]); inv = 1.0 / pv
    y = y_ev.astype(np.float64)
    for _ in range(NEWTON_IT):
        p = sigmoid(Amat @ x + b_ev)
        grad = Amat.T @ (p - y) + inv * x
        w = p * (1 - p)
        H = (Amat * w[:, None]).T @ Amat + np.diag(inv)
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            break
        x = x - step
        if np.max(np.abs(step)) < 1e-8:
            break
    return float(x[0]), x[1:]


def user_auc(y, s):
    y = np.asarray(y)
    if y.min() == y.max():
        return None
    return roc_auc_score(y, s)


def curve_asymptote(entities, E_of, b_of, users_cells, d, seed=SEED):
    """entities: id->row index into E/b. users_cells: list of (entity_ids[], labels[], order[]) per user.
    Returns per-t per-user AUC arrays + split-half asymptote per-user AUC array."""
    res = {t: [] for t in TS}
    asy = []
    rng = np.random.default_rng(seed)
    for (eids, y, order) in users_cells:
        n = len(order)
        Eall = np.array([E_of(e) for e in eids]); ball = np.array([b_of(e) for e in eids])
        # map order (positions into eids)
        for t in TS:
            tt = min(t, n)
            ev = order[:tt]; hold = np.array([p for p in range(n) if p not in set(order[:tt])])
            if len(hold) == 0 or y[hold].min() == y[hold].max():
                res[t].append(None); continue
            if tt == 0:
                alpha, k = 0.0, np.zeros(d)
            else:
                alpha, k = infer_user(Eall[ev], ball[ev], y[ev], d)
            s = ball[hold] + alpha + Eall[hold] @ k
            res[t].append(user_auc(y[hold], s))
        # split-half asymptote
        perm = rng.permutation(n); half = n // 2
        ev = perm[:half]; hold = perm[half:]
        if len(hold) and y[hold].min() != y[hold].max() and half > 0:
            alpha, k = infer_user(Eall[ev], ball[ev], y[ev], d)
            asy.append(user_auc(y[hold], ball[hold] + alpha + Eall[hold] @ k))
        else:
            asy.append(None)
    return res, asy


def mean_auc(a):
    v = [x for x in a if x is not None]
    return (float(np.mean(v)), len(v)) if v else (float("nan"), 0)


def boot_delta(a, b, seed=SEED):
    pa, pb = [], []
    for x, y in zip(a, b):
        if x is not None and y is not None:
            pa.append(x); pb.append(y)
    if not pa:
        return dict(delta=float("nan"), ci=[float("nan")] * 2, n=0)
    d = np.array(pa) - np.array(pb); rng = np.random.default_rng(seed)
    bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(BOOT)])
    return dict(delta=float(d.mean()), ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                n=int(len(d)))


def boot_mean_ci(a, seed=SEED):
    v = np.array([x for x in a if x is not None])
    if len(v) == 0:
        return [float("nan")] * 2
    rng = np.random.default_rng(seed)
    bs = np.array([v[rng.integers(0, len(v), len(v))].mean() for _ in range(BOOT)])
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


# ---------------------------------------------------- LOUO logistic MF (5-fold over users)
def train_mf(pos_pairs, n_ent, d, seed=SEED):
    """pos_pairs: list of (user_row, ent_row) POSITIVES. Uniform-negative logistic MF -> (E, b, K_train)."""
    import torch
    torch.manual_seed(seed)
    users = sorted({u for u, _ in pos_pairs}); uidx = {u: i for i, u in enumerate(users)}
    pu = np.array([uidx[u] for u, _ in pos_pairs]); pj = np.array([e for _, e in pos_pairs])
    nU = len(users); nP = len(pu)
    E = torch.zeros(n_ent, d, requires_grad=True)
    Ku = torch.zeros(nU, d, requires_grad=True)
    b = torch.zeros(n_ent, requires_grad=True)
    with torch.no_grad():
        E.normal_(0, 0.01); Ku.normal_(0, 0.01)
        # warm-start intercept at empirical entity log-rate
        cnt = np.bincount(pj, minlength=n_ent).astype(np.float64)
        rate = np.clip(cnt / max(nU, 1), 1e-3, 1 - 1e-3)
        b.copy_(torch.tensor(np.log(rate / (1 - rate)), dtype=torch.float32))
    opt = torch.optim.Adam([E, Ku, b], lr=MF_LR)
    bce = torch.nn.functional.binary_cross_entropy_with_logits
    put = torch.tensor(pu); pjt = torch.tensor(pj)
    gen = torch.Generator().manual_seed(seed)
    for ep in range(MF_EPOCHS):
        perm = torch.randperm(nP, generator=gen)
        pu2 = put[perm]; pj2 = pjt[perm]
        neg = torch.randint(0, n_ent, (nP, N_NEG), generator=gen)
        ku = Ku[pu2]
        lp = (ku * E[pj2]).sum(1) + b[pj2]
        ln = (ku.unsqueeze(1) * E[neg]).sum(2) + b[neg]
        loss = bce(lp, torch.ones_like(lp)) + bce(ln, torch.zeros_like(ln))
        loss = loss + MF_L2 * (E[pj2].pow(2).sum() + ku.pow(2).sum()) / nP
        opt.zero_grad(); loss.backward(); opt.step()
    return E.detach().numpy().astype(np.float64), b.detach().numpy().astype(np.float64)


def louo_mf_beliefs(users, cells_of, ent_universe, kthr, d=D_MF, seed=SEED):
    """5-fold over users. For each held-out fold, train MF on the OTHER users' positives, then return
    per-entity E/b to use for the held-out users' online inference. Returns E_by_user, b_by_user dicts
    keyed by user id (the embeddings valid for that user, trained without them)."""
    ent_list = sorted(ent_universe); erow = {e: i for i, e in enumerate(ent_list)}
    n_ent = len(ent_list)
    uids = [rec["u"] for rec in users]
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(uids))
    folds = [set(np.array(uids)[order[f::NFOLD]].tolist()) for f in range(NFOLD)]
    E_by, b_by = {}, {}
    for f in range(NFOLD):
        heldout = folds[f]
        pos = []
        for rec in users:
            if rec["u"] in heldout:
                continue
            urow = rec["u"]
            eid, y, _o = cells_of(rec)          # y already binarized at kthr by cells_of
            for e, yy in zip(eid, y):
                if yy >= 1:
                    pos.append((urow, erow[e]))
        # collapse to positives; train
        E, b = train_mf(pos, n_ent, d, seed=seed + f)
        for u in heldout:
            E_by[u] = E; b_by[u] = b
    return E_by, b_by, erow


def stage_b():
    t0 = time.time()
    D, split, grid, users, memb, battery = A.load_env()
    print("\n==== STAGE B -- DISCOVERY FEASIBILITY (DIRECTIONAL 173/300) ====", flush=True)
    print("PRE-REGISTERED DIRECTIONAL READS (printed before results):", flush=True)
    print("  (i)   AUC climbs with observed events t.", flush=True)
    print("  (ii)  split-half asymptote beats popularity-only by >= 0.05.", flush=True)
    print(f"  (iii) compare the TRUE-answerability asymptote to the rated-ness ceiling {A.RATEDNESS_CEILING} "
          "-- higher = discovery MORE feasible than the old arena allowed.", flush=True)

    # ---- kmap population belief (items) ----
    e = np.load(EMB); bfile = np.load(INT)
    d_km = int(e["d"]); ni = int(D["ni"])
    krow = -np.ones(ni, dtype=np.int64); krow[e["item_ids"]] = np.arange(len(e["item_ids"]))
    Ek = e["E"].astype(np.float64); bk = bfile["b"].astype(np.float64)
    pop_a = float(bfile["pop_a"]); pop_c = float(bfile["pop_c"]); logcnt = bfile["logcnt"].astype(np.float64)
    Efull = np.zeros((ni, d_km)); kept = krow >= 0; Efull[kept] = Ek[krow[kept]]
    bfull = pop_a * logcnt + pop_c; bfull[kept] = bk[krow[kept]]

    results = {"banner": "DIRECTIONAL 173/300, grid unfrozen", "channels": {}}

    # ================= ITEM channel =================
    def item_cells_of(rec, kthr=1):
        cells = {}
        for c in rec["item_llm"]:
            if c["k"] is not None:
                cells[c["j"]] = (1 if c["k"] >= kthr else 0, c["cnt"])
        for c in rec["item_data"]:
            cells.setdefault(c["j"], (1, c["cnt"]))     # data = know_well
        eids = list(cells.keys())
        y = np.array([cells[j][0] for j in eids])
        order = np.argsort([-cells[j][1] for j in eids])     # popularity desc reveal order
        return eids, y, order

    item_univ = sorted({j for rec in users for c in rec["item_llm"] for j in [c["j"]] if c["k"] is not None}
                       | {c["j"] for rec in users for c in rec["item_data"]})

    for kthr, ktag in ((1, "k>=1"), (2, "k>=2")):
        # per-user cell packs
        packs = []
        for rec in users:
            eids, y, order = item_cells_of(rec, kthr)
            packs.append((eids, y, order))
        # belief (i) popularity: b_of = bfull (kmap intercept), E_of = 0
        pop_res, pop_asy = curve_asymptote(item_univ, lambda j: np.zeros(1), lambda j: bfull[j],
                                           packs, 1)
        # belief (ii) kmap population embeddings
        km_res, km_asy = curve_asymptote(item_univ, lambda j: Efull[j], lambda j: bfull[j], packs, d_km)
        # belief (iii) LOUO judged-grid MF
        E_by, b_by, erow = louo_mf_beliefs(users, lambda r: item_cells_of(r, kthr), set(item_univ), kthr)
        mf_res = {t: [] for t in TS}; mf_asy = []
        rng = np.random.default_rng(SEED)
        for rec, (eids, y, order) in zip(users, packs):
            Eu = E_by[rec["u"]]; bu = b_by[rec["u"]]
            Emat = np.array([Eu[erow[j]] for j in eids]); bvec = np.array([bu[erow[j]] for j in eids])
            n = len(order)
            for t in TS:
                tt = min(t, n); ev = order[:tt]
                hold = np.array([p for p in range(n) if p not in set(order[:tt])])
                if len(hold) == 0 or y[hold].min() == y[hold].max():
                    mf_res[t].append(None); continue
                if tt == 0:
                    al, k = 0.0, np.zeros(D_MF)
                else:
                    al, k = infer_user(Emat[ev], bvec[ev], y[ev], D_MF)
                mf_res[t].append(user_auc(y[hold], bvec[hold] + al + Emat[hold] @ k))
            perm = rng.permutation(n); half = n // 2
            ev = perm[:half]; hold = perm[half:]
            if len(hold) and y[hold].min() != y[hold].max() and half > 0:
                al, k = infer_user(Emat[ev], bvec[ev], y[ev], D_MF)
                mf_asy.append(user_auc(y[hold], bvec[hold] + al + Emat[hold] @ k))
            else:
                mf_asy.append(None)
        results["channels"][f"item_{ktag}"] = summarize("item", ktag, pop_res, pop_asy, km_res, km_asy,
                                                         mf_res, mf_asy)

    # ================= CONCEPT channel =================
    # concept embeddings for beliefs: (ii) kmap centroid e_c = mean of member items' Efull; (iii) LOUO MF.
    def conc_cells_of(rec, kthr=1):
        cells = {}
        for c in rec["concept"]:
            if c["k"] is not None:
                cells[c["tagId"]] = (1 if c["k"] >= kthr else 0, c["pop"])
        eids = list(cells.keys())
        y = np.array([cells[t][0] for t in eids])
        order = np.argsort([-cells[t][1] for t in eids])     # member-count desc
        return eids, y, order

    conc_univ = sorted({c["tagId"] for rec in users for c in rec["concept"] if c["k"] is not None})
    # concept popularity logit (population answer-rate) + kmap centroid embeddings
    conc_rate = {}; conc_ec = {}
    tot = collections.defaultdict(lambda: [0, 0])
    for rec in users:
        for c in rec["concept"]:
            if c["k"] is not None:
                tot[c["tagId"]][0] += (1 if c["k"] >= 1 else 0); tot[c["tagId"]][1] += 1
    for t in conc_univ:
        r = np.clip(tot[t][0] / max(tot[t][1], 1), 1e-3, 1 - 1e-3); conc_rate[t] = float(np.log(r / (1 - r)))
        mems = [j for j in memb.get(str(t), []) if 0 <= j < ni and krow[j] >= 0]
        conc_ec[t] = Efull[mems].mean(0) if mems else np.zeros(d_km)

    for kthr, ktag in ((1, "k>=1"), (2, "k>=2")):
        packs = [conc_cells_of(rec, kthr) for rec in users]
        pop_res, pop_asy = curve_asymptote(conc_univ, lambda t: np.zeros(1), lambda t: conc_rate[t], packs, 1)
        km_res, km_asy = curve_asymptote(conc_univ, lambda t: conc_ec[t], lambda t: conc_rate[t], packs, d_km)
        E_by, b_by, erow = louo_mf_beliefs(users, lambda r: conc_cells_of(r, kthr), set(conc_univ), kthr)
        mf_res = {t: [] for t in TS}; mf_asy = []
        rng = np.random.default_rng(SEED)
        for rec, (eids, y, order) in zip(users, packs):
            Eu = E_by[rec["u"]]; bu = b_by[rec["u"]]
            Emat = np.array([Eu[erow[t]] for t in eids]); bvec = np.array([bu[erow[t]] for t in eids])
            n = len(order)
            for t in TS:
                tt = min(t, n); ev = order[:tt]
                hold = np.array([p for p in range(n) if p not in set(order[:tt])])
                if len(hold) == 0 or y[hold].min() == y[hold].max():
                    mf_res[t].append(None); continue
                al, k = (0.0, np.zeros(D_MF)) if tt == 0 else infer_user(Emat[ev], bvec[ev], y[ev], D_MF)
                mf_res[t].append(user_auc(y[hold], bvec[hold] + al + Emat[hold] @ k))
            perm = rng.permutation(n); half = n // 2
            ev = perm[:half]; hold = perm[half:]
            if len(hold) and y[hold].min() != y[hold].max() and half > 0:
                al, k = infer_user(Emat[ev], bvec[ev], y[ev], D_MF)
                mf_asy.append(user_auc(y[hold], bvec[hold] + al + Emat[hold] @ k))
            else:
                mf_asy.append(None)
        results["channels"][f"concept_{ktag}"] = summarize("concept", ktag, pop_res, pop_asy,
                                                           km_res, km_asy, mf_res, mf_asy)

    results["wall_min"] = round((time.time() - t0) / 60, 2)
    json.dump(results, open(OUT_JSON, "w"), indent=1, default=str)
    write_md(results)
    print(f"\n[stage B] wrote {A.OUT_MD} (Stage B) + {OUT_JSON}  (wall {results['wall_min']}m)", flush=True)
    return results


def summarize(channel, ktag, pop_res, pop_asy, km_res, km_asy, mf_res, mf_asy):
    def curve(res):
        return {t: mean_auc(res[t]) for t in TS}
    cp, ck, cm = curve(pop_res), curve(km_res), curve(mf_res)
    asy_pop = mean_auc(pop_asy); asy_km = mean_auc(km_asy); asy_mf = mean_auc(mf_asy)
    # climb: t16 vs t2 for best belief (mf)
    climb_mf = boot_delta(mf_res[16], mf_res[2]); climb_km = boot_delta(km_res[16], km_res[2])
    # asymptote beats popularity by >=0.05
    asy_pop_ci = boot_mean_ci(pop_asy)
    best_asy_name = max([("kmap", asy_km[0]), ("louo_mf", asy_mf[0])], key=lambda x: (x[1] if not np.isnan(x[1]) else -1))
    best_asy = asy_mf[0] if best_asy_name[0] == "louo_mf" else asy_km[0]
    asy_vs_pop = boot_delta(mf_asy if best_asy_name[0] == "louo_mf" else km_asy, pop_asy)
    print(f"  [B {channel}/{ktag}] pop t0={cp[0][0]:.3f} | kmap t0={ck[0][0]:.3f}->t16={ck[16][0]:.3f} "
          f"asy={asy_km[0]:.3f} | LOUO-MF t0={cm[0][0]:.3f}->t16={cm[16][0]:.3f} asy={asy_mf[0]:.3f} "
          f"| pop-asy={asy_pop[0]:.3f} | best-asy-vs-pop {asy_vs_pop['delta']:+.3f}"
          f"[{asy_vs_pop['ci'][0]:+.3f},{asy_vs_pop['ci'][1]:+.3f}] | vs 0.725: "
          f"{'ABOVE' if best_asy > A.RATEDNESS_CEILING else 'below'}", flush=True)
    return dict(pop_curve={t: cp[t][0] for t in TS}, kmap_curve={t: ck[t][0] for t in TS},
                louo_mf_curve={t: cm[t][0] for t in TS}, n_by_t={t: cm[t][1] for t in TS},
                asy_pop=asy_pop[0], asy_kmap=asy_km[0], asy_louo_mf=asy_mf[0],
                asy_pop_ci=asy_pop_ci, best_belief=best_asy_name[0], best_asy=best_asy,
                asy_vs_pop=asy_vs_pop, climb_mf_t16_vs_t2=climb_mf, climb_kmap_t16_vs_t2=climb_km,
                vs_ratedness_ceiling=dict(ceiling=A.RATEDNESS_CEILING, best_asy=best_asy,
                                          above=bool(best_asy > A.RATEDNESS_CEILING)))


def write_md(results):
    A.md_write("## STAGE B -- DISCOVERY FEASIBILITY\n\n"
               "**Pre-registered directional reads:** (i) AUC climbs with observed events t; (ii) split-half "
               f"asymptote beats popularity-only by >=0.05; (iii) compare the true-answerability asymptote "
               f"to the OLD arena's rated-ness ceiling {A.RATEDNESS_CEILING} (higher = discovery more "
               "feasible than the old arena allowed -- the headline either way).\n\n"
               "Beliefs (E5 firewall -- eval user's cells NEVER train the belief): (i) population "
               "answer-rate logit; (ii) kmap POPULATION rated-matrix embeddings (items) / member-item "
               "centroid (concepts); (iii) NEW 5-fold LEAVE-ONE-USER-OUT judged-grid logistic MF (d=16). "
               "AUC held out over unrevealed cells; needs both classes (item k>=1 is near-ceiling so many "
               "users are dropped -- n reported).\n\n")
    for chname in results["channels"]:
        r = results["channels"][chname]
        A.md_write(f"### {chname}\n\n"
                   "| belief | t=0 | t=2 | t=4 | t=8 | t=16 | split-half asymptote |\n|---|--:|--:|--:|--:|--:|--:|\n")
        for lbl, ck, asy in (("popularity (i)", r["pop_curve"], r["asy_pop"]),
                             ("kmap population (ii)", r["kmap_curve"], r["asy_kmap"]),
                             ("LOUO judged-MF (iii)", r["louo_mf_curve"], r["asy_louo_mf"])):
            A.md_write(f"| {lbl} | " + " | ".join(f"{ck[t]:.3f}" if not np.isnan(ck[t]) else " - " for t in TS)
                       + f" | {asy:.3f} |\n")
        av = r["asy_vs_pop"]; cm = r["climb_mf_t16_vs_t2"]
        A.md_write(f"\n- n users with both classes held out by t: {r['n_by_t']}.\n"
                   f"- (i) climb (LOUO-MF t16 vs t2): {cm['delta']:+.3f}[{cm['ci'][0]:+.3f},{cm['ci'][1]:+.3f}] "
                   f"(n={cm['n']}).\n"
                   f"- (ii) best belief = **{r['best_belief']}**; asymptote vs popularity "
                   f"{av['delta']:+.3f}[{av['ci'][0]:+.3f},{av['ci'][1]:+.3f}] "
                   f"({'BEATS +0.05' if av['ci'][0] >= 0.05 else ('beats pop' if av['ci'][0] > 0 else 'ties/below')}).\n"
                   f"- (iii) best asymptote {r['best_asy']:.3f} vs rated-ness ceiling "
                   f"{A.RATEDNESS_CEILING}: **{'ABOVE' if r['vs_ratedness_ceiling']['above'] else 'BELOW'}**.\n\n")
