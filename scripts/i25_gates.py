"""i25_gates.py -- the SIX pre-registered gates for the I2.5 learned fold (FABLE_AGENT_DESIGN I2.5 spec).

Runs on the 300 study users (pinned seed-123 split), the cached gate answerability grid, and the arena
NDCG@10 metric. Loads the trained fold (.cache/i25_fold_best.pt). NO LLM calls. ANY gate failure is
reported; the script does not gate Phase 4 itself (the caller decides), but prints a PASS/FAIL table.

Gates:
  G-fold1  one TRUE answer from cold HELPS, per channel (item / concept / attribute); Delta>0, CI excl 0.
  G-fold2  monotonicity in #answers 1->16, per channel and mixed (non-decreasing within noise).
  G-fold3  item-only sets >= RecVAE native fold of the SAME items.
  G-fold4  mixed item+concept >= best single channel at matched budget.
  G-fold5  smooth degradation under sigma=0.70 star noise (no cliff).
  G-fold6  full-profile fold ~= RecVAE native full-profile (ruler consistency).

Run:  python scripts/i25_gates.py
"""
import os, sys, json, time, collections
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import i25_lib as L

CKPT_BEST = ".cache/i25_fold_best.pt"
OUT_JSON = "experiments/I25_gates.json"
BOOT = 5000
SEED = 0


# ------------------------------------------------------------------ token builders (center = full-known mean)
def _cmean(known):
    return float(np.mean(list(known.values()))) if known else 0.0


def item_tokens(FR, known, ids, cmean, sigma=0.0, rng=None):
    toks, native = [], []
    for j in ids:
        r = known[j]
        if sigma > 0 and rng is not None:
            r = r + sigma * rng.standard_normal()
        toks.append((0, FR.Wn[j].numpy().astype(np.float32), float(r - cmean), f"I:{j}"))
        if known[j] >= 4:
            native.append(j)
    return toks, native


def concept_tokens(FR, D, known, k, cmean, sigma=0.0, rng=None):
    """Top-k concepts by revealed-member relevance mass over the FULL known profile; data-side aggregate."""
    its = np.array(list(known.keys()))
    if len(its) == 0:
        return []
    rel = D["concepts"]["item_tag"][its]                 # (n,200)
    rr = np.array([known[j] for j in its], float)
    if sigma > 0 and rng is not None:
        rr = rr + sigma * rng.standard_normal(len(rr))
    cr = rr - cmean
    mass = rel.sum(0)
    toks = []
    for ctag in np.argsort(-mass):
        m = float(mass[ctag])
        if m < 1e-6 or len(toks) >= k:
            break
        val = float((rel[:, ctag] * cr).sum() / m)
        toks.append((1, FR.concept_emb(int(ctag)), val, f"C:{int(ctag)}"))
    return toks


def attr_tokens(FR, D, known, k, cmean, sigma=0.0, rng=None):
    its = list(known.keys())
    rr = {j: known[j] for j in its}
    if sigma > 0 and rng is not None:
        rr = {j: known[j] + sigma * rng.standard_normal() for j in its}
    agg = collections.defaultdict(lambda: [0.0, 0.0])
    for j in its:
        crj = rr[j] - cmean
        for ak in L.item_attrs(D, j):
            agg[ak][0] += crj; agg[ak][1] += 1.0
    ordered = sorted(agg.items(), key=lambda kv: -kv[1][1])   # most-populated attrs first
    toks = []
    for ak, (s, n) in ordered[:k]:
        toks.append((2, FR.attr_emb(ak), float(s / n if n else 0.0), f"A:{ak[0]}:{ak[1]}"))
    return toks


def fold(FR, model, toks, native):
    if not toks:
        return np.zeros(FR.W.shape[1])
    return L.fold_np(FR, model, toks, native)


# ------------------------------------------------------------------ eligible users
def load_users(D, FR):
    grid = json.load(open(L.GATE_GRID))["users"]
    split = L.G.build_split(D)
    users = []
    for us, g in grid.items():
        u = int(us)
        if u not in split:
            continue
        kn, ho = split[u]; rat = dict(D["rat_by_u"][u])
        known = {j: rat[j] for j in kn if j in rat}
        prof_like = [j for j in kn if rat.get(j, 0) >= 4]
        tlike = set(j for j in ho if rat.get(j, 0) >= 4)
        if not prof_like or not tlike or len(known) < 4:
            continue
        # gate answerability for concepts/items in the per-user bank
        ans_yes = {}
        for i, (k, m) in enumerate(g["Q"]):
            key = f"C:{m['ctag']}" if k == "concept" else (f"I:{m['j']}" if k == "item" else None)
            if key:
                ans_yes[key] = L.G.is_yes({int(kk): v for kk, v in g["ans"].items()}.get(i, {}))
        users.append(dict(u=u, known=known, like=prof_like, tlike=tlike, prof=set(kn), ans=ans_yes))
    return users


def boot_ci(diff, nb=BOOT, seed=SEED):
    diff = np.asarray([d for d in diff if d is not None], float)
    if len(diff) == 0:
        return None
    rng = np.random.default_rng(seed)
    b = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(nb)])
    return dict(mean=float(diff.mean()), ci=[float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))],
                p_gt0=float((b > 0).mean()), n=len(diff))


# ================================================================== GATES
def g_fold1(FR, model, D, users):
    """One true answer from cold HELPS, per channel."""
    out = {}
    rng = np.random.default_rng(1)
    for ch in ("item", "concept", "attribute"):
        d_item = []
        for us in users:
            known = us["known"]; cm = _cmean(known)
            base = L.ndcg10(FR, np.zeros(FR.W.shape[1]), us["tlike"], us["prof"])
            if base is None:
                continue
            if ch == "item":
                # a genuinely answerable, informative single answer: a random KNOWN LIKED item
                cand = us["like"]
                j = cand[rng.integers(len(cand))]
                toks, native = item_tokens(FR, known, [j], cm)
            elif ch == "concept":
                toks = concept_tokens(FR, D, known, 1, cm); native = []
            else:
                toks = attr_tokens(FR, D, known, 1, cm); native = []
            if not toks:
                continue
            v = L.ndcg10(FR, fold(FR, model, toks, native), us["tlike"], us["prof"])
            if v is not None:
                d_item.append(v - base)
        out[ch] = boot_ci(d_item)
    return out


def g_fold2(FR, model, D, users, counts=(1, 2, 4, 8, 16)):
    """Monotonicity in #answers per channel and mixed."""
    curves = {c: {k: [] for k in counts} for c in ("item", "concept", "attribute", "mixed")}
    rng = np.random.default_rng(2)
    for us in users:
        known = us["known"]; cm = _cmean(known); likes = us["like"]
        order = list(known.keys()); rng.shuffle(order)
        like_order = [j for j in order if known[j] >= 4]
        for k in counts:
            # item: first k known items (rating-carrying)
            it_ids = order[:k]
            ti, nat = item_tokens(FR, known, it_ids, cm)
            curves["item"][k].append(L.ndcg10(FR, fold(FR, model, ti, nat), us["tlike"], us["prof"]))
            # concept: top-k concepts
            tc = concept_tokens(FR, D, known, k, cm)
            curves["concept"][k].append(L.ndcg10(FR, fold(FR, model, tc, []), us["tlike"], us["prof"]))
            # attribute: top-k attrs
            ta = attr_tokens(FR, D, known, k, cm)
            curves["attribute"][k].append(L.ndcg10(FR, fold(FR, model, ta, []), us["tlike"], us["prof"]))
            # mixed: k//2 items + k - k//2 concepts (+attrs to fill)
            ki = max(1, k // 2)
            mi, mnat = item_tokens(FR, known, order[:ki], cm)
            mc = concept_tokens(FR, D, known, k - ki, cm)
            curves["mixed"][k].append(L.ndcg10(FR, fold(FR, model, mi + mc, mnat), us["tlike"], us["prof"]))
    # summarize means + monotonicity (non-decreasing within a 0.005 noise band)
    res = {}
    for ch, ck in curves.items():
        means = [float(np.nanmean([x for x in ck[k] if x is not None])) for k in counts]
        drops = [means[i + 1] - means[i] for i in range(len(means) - 1)]
        mono = all(dd >= -0.005 for dd in drops)
        res[ch] = dict(counts=list(counts), means=[round(m, 4) for m in means],
                       min_step=round(min(drops), 4), monotone=bool(mono))
    return res


def g_fold3(FR, model, D, users, ks=(1, 2, 4, 8)):
    """Item-only learned fold >= RecVAE native fold of the SAME items."""
    rng = np.random.default_rng(3)
    res = {}
    for k in ks:
        d = []
        for us in users:
            known = us["known"]; cm = _cmean(known)
            likes = us["like"][:]; rng.shuffle(likes)
            ids = likes[:k]
            if len(ids) < min(k, 1):
                continue
            ti, nat = item_tokens(FR, known, ids, cm)
            zl = fold(FR, model, ti, nat)
            zn = FR.native_fold_np(ids)                     # native fold of the SAME liked items
            vl = L.ndcg10(FR, zl, us["tlike"], us["prof"])
            vn = L.ndcg10(FR, zn, us["tlike"], us["prof"])
            if vl is not None and vn is not None:
                d.append(vl - vn)
        res[f"k={k}"] = boot_ci(d)
    return res


def g_fold4(FR, model, D, users, budget=8):
    """Mixed item+concept >= best single channel at matched budget."""
    rng = np.random.default_rng(4)
    d_vs_best = []; detail = dict(mixed=[], item=[], concept=[])
    ki = budget // 2
    for us in users:
        known = us["known"]; cm = _cmean(known)
        order = list(known.keys()); rng.shuffle(order)
        it_ids = order[:budget]
        ti, nat = item_tokens(FR, known, it_ids, cm)
        v_item = L.ndcg10(FR, fold(FR, model, ti, nat), us["tlike"], us["prof"])
        tc = concept_tokens(FR, D, known, budget, cm)
        v_conc = L.ndcg10(FR, fold(FR, model, tc, []), us["tlike"], us["prof"])
        mi, mnat = item_tokens(FR, known, order[:ki], cm)
        mc = concept_tokens(FR, D, known, budget - ki, cm)
        v_mix = L.ndcg10(FR, fold(FR, model, mi + mc, mnat), us["tlike"], us["prof"])
        if None in (v_item, v_conc, v_mix):
            continue
        detail["mixed"].append(v_mix); detail["item"].append(v_item); detail["concept"].append(v_conc)
        d_vs_best.append(v_mix - max(v_item, v_conc))
    return dict(budget=budget, mixed_mean=round(float(np.mean(detail["mixed"])), 4),
                item_mean=round(float(np.mean(detail["item"])), 4),
                concept_mean=round(float(np.mean(detail["concept"])), 4),
                mixed_minus_best=boot_ci(d_vs_best))


def g_fold5(FR, model, D, users, sigmas=(0.0, 0.35, 0.70, 1.40), k=8):
    """Smooth degradation under star noise; no cliff (each step drop bounded, monotone)."""
    res = {}
    for s in sigmas:
        vals = []
        rng = np.random.default_rng(int(1000 + s * 100))
        for us in users:
            known = us["known"]; cm = _cmean(known)
            order = list(known.keys())
            r2 = np.random.default_rng(us["u"]); order = list(order); r2.shuffle(order)
            ids = order[:k]
            ti, nat = item_tokens(FR, known, ids, cm, sigma=s, rng=rng)
            v = L.ndcg10(FR, fold(FR, model, ti, nat), us["tlike"], us["prof"])
            if v is not None:
                vals.append(v)
        res[f"sigma={s}"] = round(float(np.mean(vals)), 4)
    ks = list(res.values())
    drops = [ks[i] - ks[i + 1] for i in range(len(ks) - 1)]
    # "no cliff": no single step loses more than 40% of the clean value
    clean = ks[0]
    cliff = any(dd > 0.40 * clean for dd in drops)
    return dict(curve=res, monotone_nonincreasing=bool(all(dd >= -0.005 for dd in drops)),
                max_step_drop=round(max(drops), 4), cliff=bool(cliff))


def g_fold6(FR, model, D, users):
    """Full-profile learned fold ~= RecVAE native full-profile fold (ruler consistency)."""
    d = []; lv = []; nv = []
    for us in users:
        known = us["known"]; cm = _cmean(known)
        # apples-to-apples: learned fold of the full known LIKED set vs native fold of the same set
        ti, nat = item_tokens(FR, {j: known[j] for j in us["like"]}, us["like"], cm)
        zl = fold(FR, model, ti, nat)
        zn = FR.native_fold_np(us["like"])                  # native full-profile (all known likes)
        vl = L.ndcg10(FR, zl, us["tlike"], us["prof"])
        vn = L.ndcg10(FR, zn, us["tlike"], us["prof"])
        if vl is not None and vn is not None:
            d.append(vl - vn); lv.append(vl); nv.append(vn)
    return dict(learned_mean=round(float(np.mean(lv)), 4), native_mean=round(float(np.mean(nv)), 4),
                delta=boot_ci(d))


# ================================================================== main
def main():
    t0 = time.time()
    print("[gates] loading data + frozen RecVAE + trained fold ...", flush=True)
    D = L.G.load_data()
    FR = L.Frozen(D)
    model = L.Fold()
    blob = torch.load(CKPT_BEST, map_location="cpu")
    model.load_state_dict(blob["model"]); model.eval()
    print(f"[gates] fold ckpt best val NDCG@10={blob['state']['best_val']:.4f} "
          f"@ep{blob['state']['best_epoch']}", flush=True)
    users = load_users(D, FR)
    print(f"[gates] {len(users)} eligible study users", flush=True)

    R = {}
    print("[gates] G-fold1 (one-answer canary) ...", flush=True)
    R["G_fold1"] = g_fold1(FR, model, D, users)
    for ch, r in R["G_fold1"].items():
        print(f"   {ch:9s} delta={r['mean']:+.4f} CI[{r['ci'][0]:+.4f},{r['ci'][1]:+.4f}] "
              f"HELPS={r['ci'][0] > 0}", flush=True)
    print("[gates] G-fold2 (monotonicity) ...", flush=True)
    R["G_fold2"] = g_fold2(FR, model, D, users)
    for ch, r in R["G_fold2"].items():
        print(f"   {ch:9s} means={r['means']} monotone={r['monotone']}", flush=True)
    print("[gates] G-fold3 (>= native, item-only) ...", flush=True)
    R["G_fold3"] = g_fold3(FR, model, D, users)
    for k, r in R["G_fold3"].items():
        print(f"   {k}: learned-native delta={r['mean']:+.4f} CI[{r['ci'][0]:+.4f},{r['ci'][1]:+.4f}]", flush=True)
    print("[gates] G-fold4 (mixed >= best single) ...", flush=True)
    R["G_fold4"] = g_fold4(FR, model, D, users)
    r = R["G_fold4"]; print(f"   mixed {r['mixed_mean']} item {r['item_mean']} concept {r['concept_mean']} "
                            f"mixed-best delta={r['mixed_minus_best']['mean']:+.4f} "
                            f"CI[{r['mixed_minus_best']['ci'][0]:+.4f},{r['mixed_minus_best']['ci'][1]:+.4f}]", flush=True)
    print("[gates] G-fold5 (noise smoothness) ...", flush=True)
    R["G_fold5"] = g_fold5(FR, model, D, users)
    print(f"   curve={R['G_fold5']['curve']} cliff={R['G_fold5']['cliff']}", flush=True)
    print("[gates] G-fold6 (ruler consistency) ...", flush=True)
    R["G_fold6"] = g_fold6(FR, model, D, users)
    r = R["G_fold6"]; print(f"   learned {r['learned_mean']} native {r['native_mean']} "
                            f"delta={r['delta']['mean']:+.4f} CI[{r['delta']['ci'][0]:+.4f},{r['delta']['ci'][1]:+.4f}]", flush=True)

    # verdicts
    v = {}
    f1 = R["G_fold1"]
    v["G_fold1"] = all(f1[c] and f1[c]["ci"][0] > 0 for c in ("item", "concept", "attribute"))
    v["G_fold2"] = all(R["G_fold2"][c]["monotone"] for c in ("item", "concept", "attribute", "mixed"))
    v["G_fold3"] = all(R["G_fold3"][k]["ci"][1] >= -0.003 for k in R["G_fold3"])  # not significantly below native
    v["G_fold4"] = R["G_fold4"]["mixed_minus_best"]["ci"][1] >= -0.003
    v["G_fold5"] = (not R["G_fold5"]["cliff"]) and R["G_fold5"]["monotone_nonincreasing"]
    v["G_fold6"] = R["G_fold6"]["delta"]["ci"][1] >= -0.005 and R["G_fold6"]["delta"]["ci"][0] <= 0.05 \
        or abs(R["G_fold6"]["delta"]["mean"]) <= 0.02
    R["verdicts"] = {k: bool(x) for k, x in v.items()}
    R["all_pass"] = bool(all(v.values()))
    R["ckpt"] = dict(best_val=blob["state"]["best_val"], best_epoch=blob["state"]["best_epoch"])
    R["n_users"] = len(users)
    R["wall_min"] = round((time.time() - t0) / 60, 2)

    os.makedirs("experiments", exist_ok=True)
    json.dump(R, open(OUT_JSON, "w"), indent=1, default=str)
    print("\n==== GATE VERDICTS ====", flush=True)
    for k, x in R["verdicts"].items():
        print(f"   {k}: {'PASS' if x else 'FAIL'}", flush=True)
    print(f"   ALL PASS: {R['all_pass']}  (wall {R['wall_min']}m)  -> {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
