"""kmap_build.py -- LEARNED answerability/knowledge model ("K-map"), PART 1: population-scale
ITEM KNOWLEDGE EMBEDDINGS by logistic matrix factorization of the binary KNOWN matrix.

MODEL:  P(u knows j) = sigmoid( b_j + k_u . e_j )
  b_j = item intercept (popularity baseline; recovers the t=0 / popularity-only predictor)
  e_j = item knowledge embedding (d-dim; the collaborative "who-knows-what" geometry)
  k_u = per-user knowledge vector (fit here for TRAINING users only; DISCARDED -- study users
        re-infer their own k_u online in kmap_validate.py from observed answer/refusal events).

DATA (zero leak, by construction): the binary KNOWN matrix of ML-25M TRAINING users (trU),
  KNOWN[u,j] = 1 iff user u rated item j. trU is disjoint from the eval split (va/te); the 298/300
  study users are te-based, so NO study-user row can enter training (asserted). Items restricted to
  those with >= MIN_RATERS raters (stability). Training users optionally downsampled for speed
  (study users NEVER downsampled -- they are not in trU at all).

RECIPE: logistic MF trained by minibatch SGD (Adam) with uniform negative sampling over the kept-item
  universe; deterministic seed. A small population popularity-logit fallback (a*log(cnt+1)+c) is fit so
  that items outside the kept universe still receive a sane intercept in validation.

ARTIFACTS -> .cache/instrument2/:
  kmap_emb.npz        : item_ids (dense), E (K,d), d, config
  kmap_intercepts.npz : item_ids (dense), b (K,), pop_a, pop_c (log-cnt fallback), base_logit
  kmap_meta.json      : recipe + coverage of the 160-item probe bank and the 1716-item judged bank

NO LLM calls. Run:  python scripts/kmap_build.py
"""
import os, sys, json, time, collections
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "instrument2"))

META = "data/movielens/.cache/ml25m/meta.npz"
GATE_GRID = ".cache/instrument2/answerability_grid_ml25m.json"
MAIN_GRID = ".cache/instrument2/answerability_mainstudy_grid.json"
OUT_EMB = ".cache/instrument2/kmap_emb.npz"
OUT_INT = ".cache/instrument2/kmap_intercepts.npz"
OUT_META = ".cache/instrument2/kmap_meta.json"

# ---- hyper-parameters (deterministic; documented) ----
D_EMB = 16
MIN_RATERS = 20
NU_TRAIN = 40000          # downsample training users for tractable CPU MF (study users NOT in trU)
EPOCHS = 3
BATCH = 200000
N_NEG = 3
LR = 0.05
L2_EMB = 1e-5             # weight decay on e_j, k_u (b_j unregularized so it tracks popularity)
SEED = 0


def study_user_ids():
    """The study-user dense ids present in the cached grids (te-based; disjoint from trU)."""
    ids = set()
    for gp in (GATE_GRID, MAIN_GRID):
        if os.path.exists(gp):
            g = json.load(open(gp))
            ids |= {int(u) for u in g["users"].keys()}
    return ids


def probe_bank_and_judged():
    """Rebuild the 160-item top-coverage probe bank (as i25_phase4_fair.build_universe) + the set of
    all LLM-judged item ids (union of gate + main grids) = the 'judged bank'. Uses the study users'
    known halves for coverage. Returns (bank_ids, judged_ids)."""
    import llm_answerability_gate as G
    D = G.load_data()
    split = G.build_split(D)
    gate = json.load(open(GATE_GRID))["users"]
    # replicate assemble's study-user filter to get the SAME 298 cohort + their known sets
    cover = collections.Counter()
    judged = set()
    for us in gate:
        u = int(us)
        if u not in split:
            continue
        kn, ho = split[u]
        rat = dict(D["rat_by_u"][u])
        known = {j: rat[j] for j in kn if j in rat}
        like = [j for j in kn if rat.get(j, 0) >= 4]
        tlike = set(j for j in ho if rat.get(j, 0) >= 4)
        if not like or not tlike or len(known) < 4:
            continue
        for j in known:
            cover[j] += 1
    bank = [j for j, c in cover.most_common() if c >= 3][:160]
    # judged item ids across both grids
    for gp in (GATE_GRID, MAIN_GRID):
        g = json.load(open(gp))["users"]
        for us, rec in g.items():
            for k, m in rec["Q"]:
                if k in ("item", "valid_rated", "valid_never") and "j" in m:
                    judged.add(int(m["j"]))
    return set(int(j) for j in bank), judged


def main():
    import torch
    t0 = time.time()
    print("[kmap] loading ML-25M meta ...", flush=True)
    d = np.load(META)
    uu, ii = d["uu"].astype(np.int64), d["ii"].astype(np.int64)
    cnt = d["cnt"].astype(np.float64)
    ni = int(d["ni"])
    trU = d["trU"].astype(np.int64)
    study = study_user_ids()
    leak = len(set(trU.tolist()) & study)
    assert leak == 0, f"LEAK: {leak} study users in trU"
    print(f"[kmap] ni={ni}, |trU|={len(trU)}, |study|={len(study)} (leak={leak})", flush=True)

    # ---- downsample training users (deterministic) ----
    rng = np.random.default_rng(SEED)
    if NU_TRAIN < len(trU):
        keepU = np.sort(rng.choice(trU, size=NU_TRAIN, replace=False))
        note_ds = f"downsampled to {NU_TRAIN} of {len(trU)} trU users (seed {SEED})"
    else:
        keepU = np.sort(trU); note_ds = f"all {len(trU)} trU users"
    keepU_set = set(keepU.tolist())
    print(f"[kmap] {note_ds}", flush=True)

    # ---- select training rows (trU sample only) ----
    print("[kmap] selecting training interactions ...", flush=True)
    umask = np.isin(uu, keepU)
    us = uu[umask]; it = ii[umask]
    print(f"[kmap] {len(us)} interactions from {len(keepU)} users", flush=True)

    # ---- item filter: >= MIN_RATERS raters among the selected population (zero-leak recount) ----
    raters = np.bincount(it, minlength=ni)
    kept_items = np.where(raters >= MIN_RATERS)[0]
    K = len(kept_items)
    new_j = -np.ones(ni, dtype=np.int64)
    new_j[kept_items] = np.arange(K)
    print(f"[kmap] kept {K} of {ni} items (>= {MIN_RATERS} raters)", flush=True)

    # remap positives to kept-item space; drop rows on filtered items
    jj = new_j[it]
    good = jj >= 0
    # compress users to contiguous rows
    uniqU, urow = np.unique(us[good], return_inverse=True)
    pos_u = urow.astype(np.int64)
    pos_j = jj[good].astype(np.int64)
    nU = len(uniqU); nP = len(pos_j)
    print(f"[kmap] positives={nP} over {nU} users x {K} items", flush=True)

    # popularity-logit fallback a*log(cnt+1)+c fit to per-item empirical known-rate (for uncovered items)
    logc_all = np.log(cnt + 1.0)
    emp_rate = raters / float(nU)                     # fraction of training users who rated each item
    emp_rate = np.clip(emp_rate, 1e-6, 1 - 1e-6)
    y_lin = np.log(emp_rate / (1 - emp_rate))         # logit of empirical rate
    m_fit = raters >= MIN_RATERS
    A = np.column_stack([logc_all[m_fit], np.ones(m_fit.sum())])
    coef, *_ = np.linalg.lstsq(A, y_lin[m_fit], rcond=None)
    pop_a, pop_c = float(coef[0]), float(coef[1])
    base_logit = float(np.log(emp_rate[m_fit].mean() / (1 - emp_rate[m_fit].mean())))
    print(f"[kmap] popularity-logit fallback: a={pop_a:.4f} c={pop_c:.4f}; base_logit={base_logit:.4f}", flush=True)

    # ---- logistic MF (torch, Adam, uniform negatives) ----
    torch.manual_seed(SEED)
    E = torch.zeros(K, D_EMB, requires_grad=True)
    Ku = torch.zeros(nU, D_EMB, requires_grad=True)
    b = torch.zeros(K, requires_grad=True)
    with torch.no_grad():
        E.normal_(0, 0.01); Ku.normal_(0, 0.01)
        b.copy_(torch.tensor(y_lin[kept_items], dtype=torch.float32))   # warm-start intercepts at popularity logit
    opt = torch.optim.Adam([E, Ku, b], lr=LR)
    bce = torch.nn.functional.binary_cross_entropy_with_logits

    pos_u_t = torch.tensor(pos_u); pos_j_t = torch.tensor(pos_j)
    gen = torch.Generator().manual_seed(SEED)
    nbatch = (nP + BATCH - 1) // BATCH
    for ep in range(EPOCHS):
        perm = torch.randperm(nP, generator=gen)
        tot = 0.0
        for bi in range(nbatch):
            sl = perm[bi * BATCH:(bi + 1) * BATCH]
            pu = pos_u_t[sl]; pj = pos_j_t[sl]
            nn = len(sl)
            neg_j = torch.randint(0, K, (nn, N_NEG), generator=gen)
            # logits
            ku = Ku[pu]                                   # (nn,d)
            lp = (ku * E[pj]).sum(1) + b[pj]              # positives
            ln = (ku.unsqueeze(1) * E[neg_j]).sum(2) + b[neg_j]   # (nn,N_NEG)
            loss = bce(lp, torch.ones_like(lp)) + bce(ln, torch.zeros_like(ln))
            loss = loss + L2_EMB * (E[pj].pow(2).sum() + ku.pow(2).sum() + E[neg_j].pow(2).sum()) / nn
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * nn
        print(f"[kmap] epoch {ep+1}/{EPOCHS} mean-loss {tot/nP:.4f} [{(time.time()-t0)/60:.1f}m]", flush=True)

    E_np = E.detach().numpy().astype(np.float32)
    b_np = b.detach().numpy().astype(np.float32)

    # ---- coverage report ----
    bank, judged = probe_bank_and_judged()
    kept_set = set(kept_items.tolist())
    bank_cov = len(bank & kept_set) / max(len(bank), 1)
    judged_cov = len(judged & kept_set) / max(len(judged), 1)
    print(f"[kmap] coverage: probe-bank {len(bank & kept_set)}/{len(bank)} ({bank_cov:.3f}); "
          f"judged-bank {len(judged & kept_set)}/{len(judged)} ({judged_cov:.3f})", flush=True)

    # ---- persist ----
    os.makedirs(os.path.dirname(OUT_EMB), exist_ok=True)
    np.savez(OUT_EMB, item_ids=kept_items.astype(np.int64), E=E_np, d=D_EMB)
    np.savez(OUT_INT, item_ids=kept_items.astype(np.int64), b=b_np,
             pop_a=pop_a, pop_c=pop_c, base_logit=base_logit, logcnt=logc_all.astype(np.float32))
    meta = dict(
        recipe=dict(model="P(u knows j)=sigmoid(b_j + k_u . e_j)", d=D_EMB, min_raters=MIN_RATERS,
                    nu_train=int(nU), n_pos=int(nP), n_items_kept=int(K), n_items_total=ni,
                    epochs=EPOCHS, batch=BATCH, n_neg=N_NEG, lr=LR, l2_emb=L2_EMB, seed=SEED,
                    downsample=note_ds, neg_sampling="uniform over kept items",
                    warm_start="b_j initialized to empirical popularity logit"),
        zero_leak=dict(study_users=len(study), leak_into_trU=leak),
        popularity_fallback=dict(pop_a=pop_a, pop_c=pop_c, base_logit=base_logit,
                                 desc="logit(P) ~= pop_a*log(cnt+1)+pop_c for items outside the kept universe"),
        coverage=dict(probe_bank_n=len(bank), probe_bank_covered=len(bank & kept_set), probe_bank_frac=bank_cov,
                      judged_bank_n=len(judged), judged_bank_covered=len(judged & kept_set),
                      judged_bank_frac=judged_cov),
        artifacts=dict(emb=OUT_EMB, intercepts=OUT_INT),
        wall_min=round((time.time() - t0) / 60, 2))
    json.dump(meta, open(OUT_META, "w"), indent=1)
    print(f"[kmap] wrote {OUT_EMB}, {OUT_INT}, {OUT_META}  (wall {meta['wall_min']}m)", flush=True)


if __name__ == "__main__":
    main()
