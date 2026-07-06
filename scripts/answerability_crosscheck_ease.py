"""CROSS-CHECK A (Hole 1, shared-prior circularity): independent-CF value predictor.

Bank-restricted EASE (NOT the full 18,430-item catalogue -> memory). Train EASE item-item
weights on the KNOWN-portion ratings of ALL users, restricted to a bounded item universe
(top-popularity items UNION the masked target items => guaranteed coverage, a few thousand items
=> trivial dense inverse). Predict the SAME masked rated items the LLM predicted (per gate user,
8 known-half items masked out), using each user's OTHER known-half ratings as context.

CONSTRAINTS enforced:
  - EASE never sees any user's HELD-OUT half (the recommendation targets, seed-123 split). Gate
    users' held-out interactions are dropped from the training matrix.
  - The masked items are in the KNOWN half (verified 2400/2400). At PREDICT time we build each
    gate user's context vector from known-half MINUS the masked set, mirroring exactly what the
    LLM was shown (SYS_MASK profile = known-half minus masked).
  - Diagonal-zeroed EASE weights: a user's score for item i never uses their own rating of i.

PASS = EASE produces sane predictions (MAE < ~1.0 star, positive pred/true corr) so we HAVE a
second value predictor to report sensitivity against. This does NOT gate the adaptive claim
(value is sensitivity-only per the Hole-1 rule); it closes the shared-prior circularity by
providing a CF predictor that shares no text-source prior with the LLM judge.

Compares EASE MAE/corr vs the LLM's masked MAE (0.68 stars / corr 0.50, n=2400) on the SAME set.
"""
import os, json, argparse, collections
import numpy as np
import scipy.sparse as sp

META = "data/movielens/.cache/ml25m/meta.npz"
SPLIT_CACHE = ".cache/instrument2/answerability_answerer_split.json"
GRID_CACHE = ".cache/instrument2/answerability_grid_ml25m.json"
OUT_JSON = "experiments/answerability_crosscheck_ease.json"

ANSWERER_SPLIT_SEED = 123
TOPK_POPULAR = 4000     # bank universe base (top-popularity items)
LAMBDA = 500.0          # EASE L2 (standard ML-scale value)


def build_ease():
    d = np.load(META)
    uu = d["uu"].astype(np.int64); ii = d["ii"].astype(np.int64); rr = d["rr"].astype(np.float64)
    cnt = d["cnt"].astype(np.float64); ni = int(d["ni"]); nu = int(d["nu"])

    split = json.load(open(SPLIT_CACHE))["split"]              # per gate-family user: known/heldout dense ids
    grid = json.load(open(GRID_CACHE))["users"]               # 300 study users w/ masked preds
    gate_users = sorted(int(u) for u in grid)

    # ---- masked targets (the SAME set the LLM predicted) ----
    masked_rows = []   # (user, dense_item, true_rating)
    for u in gate_users:
        for m in grid[str(u)]["mask"]:
            masked_rows.append((u, int(m["j"]), float(m["true"])))
    masked_items = set(j for _, j, _ in masked_rows)

    # ---- bank universe = top-popularity items UNION masked targets (guarantee coverage) ----
    top_pop = set(int(j) for j in np.argsort(-cnt)[:TOPK_POPULAR])
    universe = sorted(top_pop | masked_items)
    uni_index = {j: k for k, j in enumerate(universe)}
    nU = len(universe)
    print(f"[ease] universe={nU} items (top{TOPK_POPULAR} pop + {len(masked_items)} masked; "
          f"masked already in pop: {len(masked_items & top_pop)})", flush=True)

    # ---- drop gate users' HELD-OUT interactions from the training matrix (never train on targets)
    heldout_codes = set()
    for u in gate_users:
        for j in split[str(u)]["heldout"]:
            heldout_codes.add(int(u) * ni + int(j))
    codes = uu * ni + ii
    drop = np.isin(codes, np.fromiter(heldout_codes, np.int64, len(heldout_codes)))
    print(f"[ease] dropping {int(drop.sum())} gate-user held-out interactions from training", flush=True)

    # ---- restrict to universe items + not-dropped ----
    in_uni = np.zeros(ni, bool)
    in_uni[universe] = True
    keep = in_uni[ii] & (~drop)
    uu_k = uu[keep]; rr_k = rr[keep]
    ii_k = np.array([uni_index[int(j)] for j in ii[keep]], np.int64)
    print(f"[ease] training interactions in universe: {len(uu_k)} over {len(np.unique(uu_k))} users", flush=True)

    # ---- binary interaction matrix M (users x universe) + item means from ratings ----
    M = sp.csr_matrix((np.ones(len(uu_k)), (uu_k, ii_k)), shape=(nu, nU))
    M.data[:] = 1.0  # binarize (dedupe multi-interactions)
    Rsum = sp.csr_matrix((rr_k, (uu_k, ii_k)), shape=(nu, nU))
    item_cnt = np.asarray(M.sum(0)).ravel()
    item_rsum = np.asarray(Rsum.sum(0)).ravel()
    mu = np.divide(item_rsum, np.maximum(item_cnt, 1.0))       # per-item mean rating (universe)
    global_mu = float(rr_k.mean())
    mu[item_cnt == 0] = global_mu

    # ---- EASE closed form on binary M: B = -P/diag(P), diag(B)=0 ----
    print("[ease] computing Gram + inverse ...", flush=True)
    G = (M.T @ M).toarray().astype(np.float64)                # (nU,nU)
    G[np.diag_indices_from(G)] += LAMBDA
    P = np.linalg.inv(G)
    diagP = np.diag(P).copy()
    B = -P / diagP[None, :]
    np.fill_diagonal(B, 0.0)
    del G, P

    # ---- predict masked ratings: pred_ui = mu_i + sum_j (r_uj - mu_j) * B[j,i] ----
    # context = user's KNOWN-half ratings MINUS the masked set (exactly what the LLM saw).
    from collections import defaultdict
    user_known_ratings = {}  # u -> dict(dense_j -> rating) restricted to universe
    grid_masked_by_u = defaultdict(set)
    for u, j, _ in masked_rows:
        grid_masked_by_u[u].add(j)
    # need each gate user's known-half ratings; reuse full rating dict from meta for those users
    want = set(gate_users)
    ratmap = defaultdict(dict)
    selu = np.isin(uu, np.fromiter(want, np.int64, len(want)))
    for u, j, r in zip(uu[selu].tolist(), ii[selu].tolist(), rr[selu].tolist()):
        ratmap[u][j] = float(r)

    rows_out = []
    covered = skipped = 0
    for u, j_target, true_r in masked_rows:
        if j_target not in uni_index:
            skipped += 1; continue
        knownset = set(int(x) for x in split[str(u)]["known"])
        masked_set = grid_masked_by_u[u]
        # context vector over universe: known-half minus masked, ratings centered
        rc = np.zeros(nU)
        for jj in knownset:
            if jj in masked_set:            # exclude ALL masked (mirror LLM profile)
                continue
            if jj in uni_index and jj in ratmap[u]:
                rc[uni_index[jj]] = ratmap[u][jj] - mu[uni_index[jj]]
        col = uni_index[j_target]
        pred = mu[col] + float(rc @ B[:, col])
        pred = float(np.clip(pred, 0.5, 5.0))
        rows_out.append(dict(user=u, item=j_target, true=true_r, pred=pred))
        covered += 1

    errs = np.array([abs(r["pred"] - r["true"]) for r in rows_out])
    tp = np.array([[r["true"], r["pred"]] for r in rows_out])
    mae = float(errs.mean())
    corr = float(np.corrcoef(tp[:, 0], tp[:, 1])[0, 1]) if tp[:, 1].std() > 1e-9 else float("nan")
    rmse = float(np.sqrt((errs ** 2).mean()))

    # naive baselines for context
    gm_mae = float(np.mean(np.abs(tp[:, 0] - global_mu)))
    itemmean_pred = np.array([mu[uni_index[r["item"]]] for r in rows_out])
    im_mae = float(np.mean(np.abs(tp[:, 0] - itemmean_pred)))

    result = dict(
        method="Bank-restricted EASE (top-%d popular UNION masked targets = %d-item universe; "
               "lambda=%.0f; binary M item-item weights; item-mean-centered rating prediction). "
               "Trained on KNOWN-portion ratings of all users; gate-user held-out interactions "
               "dropped; predicted the SAME masked known-half items the LLM predicted, context = "
               "known-half minus masked (mirrors the LLM profile)." % (TOPK_POPULAR, nU, LAMBDA),
        universe_items=nU, lambda_l2=LAMBDA, n_predicted=covered, n_skipped_out_of_universe=skipped,
        ease_mae=mae, ease_pred_true_corr=corr, ease_rmse=rmse,
        global_mean_mae=gm_mae, item_mean_mae=im_mae, global_mu=global_mu,
        llm_mae_reference=0.6789583333333333, llm_corr_reference=0.5004255779445698, llm_n=2400,
        pass_sane=bool(mae < 1.0 and (corr == corr and corr > 0)),
    )
    os.makedirs("experiments", exist_ok=True)
    json.dump(result, open(OUT_JSON, "w"), indent=1)
    print(json.dumps({k: v for k, v in result.items() if k != "method"}, indent=1), flush=True)
    print(f"\n[VERDICT] EASE MAE={mae:.3f} corr={corr:.3f} (n={covered}) vs LLM MAE=0.679 corr=0.500 "
          f"-> {'PASS (sane 2nd predictor)' if result['pass_sane'] else 'FAIL'}", flush=True)
    print(f"[saved] {OUT_JSON}", flush=True)
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.parse_args()
    build_ease()
