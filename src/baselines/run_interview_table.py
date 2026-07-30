r"""run_interview_table.py -- THE INTERVIEW TABLE: six lit-anchored strategies x the elicitation-lineage
recommenders + our instrument, at matched question budgets.

THE ARGUMENT THIS TESTS. Elicitation papers evaluate their question-selection strategies through their
own recommenders, which are weak by construction (a shrunk group mean, a ridge seed, a Gaussian belief
over a frozen basis). If the ranking of STRATEGIES depends on which RECOMMENDER consumes the answers,
then those papers' conclusions are partly artifacts of a weak instrument -- and the field has been
measuring elicitation through a lens that distorts it. Rashid's published ordering
(helf > entropy0 > popularity > pure_entropy > random) gives us a falsifiable target: it either
replicates on a given recommender or inverts, per pair, with a paired CI.

RECOMMENDERS, each on its own published input contract:
  ours            set encoder, GRADED + SIGNED answers folded natively
  recvae_likes    strong binary control -- gets only the LIKES among the asked items (a bag-of-items
                  vector has no slot for a negative); shows what the strategy ranking looks like
                  through a strong recommender that cannot hear a "no"
  golbandi_leaf   the WSDM'11 node model: shrunk mean profile of train users with the same
                  like/dislike/unknown pattern. k<=8 only (see golbandi_leaf.MAX_DEPTH)
  rbmf            ridge seed onto a frozen basis, signed targets
  belief_mf       conjugate Gaussian belief, signed observations
  mostpop         the PRIOR: ignores the answers entirely, so it is ONE number, not a curve --
                  a horizontal reference line, evaluated once

PROTOCOL. Candidate pool = the arm-N protocol pool (every rated fold-in-side item, likes AND dislikes),
FIXED and strategy-independent. This deliberately departs from strategy_ladder's credit-neutral
convention of masking every ASKED item: if the pool depended on what a strategy asked, a strategy that
asks about popular items would delete those items from its own candidate pool, systematically
penalising the popularity arm -- a confound landing on exactly the arm the literature ranks third.
An asked-but-unrated item still BURNS the question (the lit protocol); it simply stays rankable.

  python src/baselines/run_interview_table.py [--budgets 1,2,4,8,16] [--only ours,rbmf]
"""
import os
import sys
import json
import time
import argparse
import numpy as np
import torch
from scipy import sparse

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_ROOT, "src", "instrument"))

import metrics as M
import rbmf_seed, belief_mf, pop, recvae
from arm_n import load_arm_n
from arm_n_tower import SNAP_DEFAULT
import interview_strategies as ST
from golbandi_leaf import GolbandiLeaf, MAX_DEPTH
from train_tower_t2 import build_model, make_graded_predict_fn, compute_head_mask, level_to_sv

OUT = os.path.join(_ROOT, "experiments", "baselines", "interview")
RECS = ["ours", "recvae_likes", "golbandi_leaf", "rbmf", "belief_mf"]
GLOBAL_ARMS = {"popularity", "pure_entropy", "entropy0", "helf"}


def logln(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "run.log"), "a") as f:
        f.write(line + "\n")


# ------------------------------------------------------------------ answer -> per-contract input
def answers_to_csr(answered, n_users, n_items, mode):
    """mode='levels'  -> level+1 (the tower's graded encoding)
       mode='signed'  -> level_to_sv: +1 at 5.0 stars, 0 at 2.75, -1 at 0.5 (RBMF / belief-MF targets)
       mode='likes'   -> 1.0 for level>=7 (r>=4.0) only; dislikes are DROPPED, which is precisely what
                         a binary recommender's contract can express."""
    rows, cols, vals = [], [], []
    for u, toks in enumerate(answered):
        for (i, lv) in toks:
            if mode == "likes":
                if lv < 7:
                    continue
                v = 1.0
            elif mode == "levels":
                v = float(lv) + 1.0
            else:
                v = float(level_to_sv(lv))
            rows.append(u); cols.append(i); vals.append(v)
    return sparse.csr_matrix((np.asarray(vals, np.float32), (rows, cols)),
                             shape=(n_users, n_items), dtype=np.float32)


def cursor_predict(score_matrix_fn, batch=500):
    """Wrap a function that produces scores for a row RANGE into metrics.evaluate's predict_fn(X)
    contract. evaluate walks rows in order with a fixed batch size, so a cursor is safe; it is reset
    before every evaluate call (same pattern SASRec uses for its timestamp-ordered split)."""
    state = {"cursor": 0}

    def predict(X):
        b = X.shape[0]
        s = state["cursor"]
        out = score_matrix_fn(s, s + b)
        state["cursor"] = s + b
        return out
    predict.reset = lambda: state.update(cursor=0)
    return predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budgets", default="1,2,4,8,16")
    ap.add_argument("--only", default=None)
    ap.add_argument("--arms", default=None)
    ap.add_argument("--snapshot", default=SNAP_DEFAULT)
    a = ap.parse_args()
    budgets = [int(x) for x in a.budgets.split(",")]
    recs = RECS if not a.only else [r for r in RECS if r in set(a.only.split(","))]
    arms = ST.ARMS if not a.arms else [s for s in ST.ARMS if s in set(a.arms.split(","))]

    D = load_arm_n(log=logln)
    ni, n = D["n_items"], D["te_tr"].shape[0]
    pool, head_mask, te_te = D["pool"], D["head_mask"], D["te_te"]
    _hm, cnt = compute_head_mask(D["train"], ni)

    glob, bank = ST.build_orders(D["g_train"], D["train"], ni, log=logln)
    lvl = ST.level_lookup(D["g_te_tr"])
    logln(f"[interview] {len(recs)} recommenders x {len(arms)} strategies x {len(budgets)} budgets")

    # ---------------------------------------------------------------- recommenders
    built = {}
    if "ours" in recs:
        ma = argparse.Namespace(arch="i25", teacher="warm_init", t_hidden=600, t_latent=200,
                                token="film", train_decoder=False, sign_prior=True, unfreeze_emb=False,
                                lr=3e-4, warm_lr_scale=0.1, full_kd=False, full_kd_w=0.3)
        enc, dec, _t, _p, _g = build_model(ma, ni, cnt)
        blob = torch.load(a.snapshot, map_location="cpu")
        enc.load_state_dict(blob["enc"]); dec.load_state_dict(blob["decoder"]); enc.eval()
        built["ours"] = ("levels", (enc, dec.weight.detach(), dec.bias.detach()))
    if "recvae_likes" in recs:
        ck = os.path.join(_ROOT, ".cache", "baselines", "recvae_ml25m_liang.pt")
        ra = recvae._defaults()
        ra.epochs = int(torch.load(ck, map_location="cpu")["state"]["epoch"]); ra.max_minutes = 0.0
        built["recvae_likes"] = ("likes", recvae.fit(D["train"], ni, args=ra, ckpt=ck, log=logln))
    if "rbmf" in recs:
        built["rbmf"] = ("signed", rbmf_seed.fit(D["train"], ni, log=logln))
    if "belief_mf" in recs:
        built["belief_mf"] = ("signed", belief_mf.fit(D["train"], ni, log=logln))
    if "golbandi_leaf" in recs:
        built["golbandi_leaf"] = ("raw", GolbandiLeaf(D["g_train"], ni, lam=8.0, log=logln))

    results = {}
    outp = os.path.join(OUT, "interview_table.json")

    # ---------------------------------------------------------------- MostPop: one number
    pr_pop = pop.fit(D["train"], ni, log=logln)
    r = M.evaluate(pr_pop, D["te_tr"], te_te, batch_size=500, head_mask=head_mask, mask_X=pool)
    results["mostpop"] = {"_prior": {"full": r["ndcg@10"], "tail": r["tail_ndcg@10"]}}
    logln(f"[interview] mostpop PRIOR (constant, answer-independent): "
          f"full={r['ndcg@10']:.4f} tail={r['tail_ndcg@10']:.4f}")

    for name in recs:
        mode, obj = built[name]
        results[name] = {}
        for arm in arms:
            results[name][arm] = {}
            for k in budgets:
                if name == "golbandi_leaf" and k > MAX_DEPTH:
                    logln(f"[interview] SKIP golbandi_leaf k={k} > {MAX_DEPTH}: 3^k answer patterns "
                          f"fragment the leaves to nothing and the shrunk mean returns the global "
                          f"mean -- that is Most-Popular in disguise, not the method.")
                    continue
                ts = time.time()
                asked, answered = ST.ask(arm, glob, bank, n, ni, lvl, k)
                n_ans = float(np.mean([len(t) for t in answered]))
                if mode == "raw":
                    ga = asked[0] if arm in GLOBAL_ARMS else None
                    sc = obj.scores(asked, answered, global_asked=ga)
                    predict = cursor_predict(lambda s, e, _sc=sc: _sc[s:e].copy())
                    predict.reset()
                    res = M.evaluate(predict, D["te_tr"], te_te, batch_size=500,
                                     head_mask=head_mask, mask_X=pool)
                else:
                    Xi = answers_to_csr(answered, n, ni, mode)
                    if name == "ours":
                        e_, W_, b_ = obj
                        pr = make_graded_predict_fn(e_, W_, b_, Xi, check_nnz=False)
                    else:
                        pr = obj
                    res = M.evaluate(pr, Xi, te_te, batch_size=500, head_mask=head_mask, mask_X=pool)
                results[name][arm][f"k{k}"] = {"full": res["ndcg@10"], "tail": res["tail_ndcg@10"],
                                               "answered": n_ans}
                logln(f"[interview] {name:14s} {arm:14s} k={k:2d} answered={n_ans:4.1f}/{k} "
                      f"full={res['ndcg@10']:.4f} tail={res['tail_ndcg@10']:.4f} "
                      f"({(time.time() - ts) / 60:.1f}m)")
                json.dump({"lit_rank": ST.LIT_RANK, "cites": ST.CITES, "budgets": budgets,
                           "results": results}, open(outp, "w"), indent=2)
    logln(f"[interview] done -> {outp}")


if __name__ == "__main__":
    main()
