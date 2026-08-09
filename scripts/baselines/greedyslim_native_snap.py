r"""greedyslim_native_snap.py -- SNAP-4: Greedy SLIM (Proissl, Vatic & Waldschmidt, arXiv:2406.06061)
NATIVE snap in the authors' own setup, plus the asked-item-exclusion ABLATION.

===============================================================================================
STEP 0 -- THE PUBLISHED TARGET, WITH SOURCE                                                [V]
===============================================================================================
Claudius Proissl, Amel Vatic, Helmut Waldschmidt, "Greedy SLIM: A SLIM-Based Approach For
Preference Elicitation", arXiv:2406.06061 [cs.IR], 10 Jun 2024.  **CoRR only -- dblp lists no
venue.**  Provenance caveat carried in every report: arXiv-only preprint; its user study is
n=103 with "many of them were somehow linked to the authors".

Table 3 (ML-25M), NDCG@10, questionnaire size k:

    k        |   5      10      15      20
    ---------+--------------------------------
    ALL  QGSLIM  | .3390   .3594   .3709   .3752      <- static, prefix-optimal
    ALL  QBandit | .3382   .3456   .3221   .2916      <- adaptive (Christakopoulou KDD'16)
    TAIL QGSLIM  | .1163   .1543   .1674   .1779
    TAIL QBandit | .1183   .1465   .1488   .1466

(Netflix goes the other way at k<=10: QBandit .3030/.3251 vs QGSLIM .2859/.3042.)

THEIR CONTRACT, verbatim where it matters -- NONE OF IT IS OUR CONTRACT, so these numbers are
NOT orderable against our 0.3496/0.2491 Liang cells.  The snap happens entirely inside their
setup.
  * ML-25M **UNFILTERED**: "We do not filter these sets to ensure that we do not accidentally
    introduce any biases."  162,541 users / 59,047 items / 25,000,095 ratings.
  * Random **90/10 user split**; |U_test| = 0.1|U|.
  * **Explicit 1-5 ratings, not binarised.**  "the values x_ui of X are either 0 if user u has
    not rated item i yet or x_ui = r_ui."
  * Graded gain: "We set the gain g_ui of user u for item i to be **2^{x_ui} - 1** in order to
    emphasize that a high rating is much more important than a medium rating."
  * "The NDCG is the DCG normalized by the **maximum DCG among all subsets I subset I_R**
    (for a fixed user u)."  -> **IDCG IS COMPUTED OVER THE CANDIDATE SET I_R.**
  * Candidates: "we do not apply any kind of item sampling to avoid the problems reported in
    Krichene and Rendle"; candidates are **all items with x_ui = 0** (i.e. everything the
    questionnaire has not revealed a rating for).
  * S3.3.2: "Let I_Q be the set of items asked by the questionnaire and let I_R be the set of
    recommended items.  **We always require I_Q ∩ I_R = ∅ to avoid trivial recommendations.**"
  * Long tail, verbatim ours: "Following Cremonesi et al. (2010), we refer to the set of most
    popular items that represent **33% of the ratings** as short-head items. The remaining
    items are called long-tail items... the **top-N recommendation is restricted to long-tail
    items**."
  * Hyperparameters ML-25M: **lambda_1 = 2^12, lambda_F = 2^16** (grid over powers of two).
  * Budgets k in {5,10,15,20}.  Cost: "19 minutes per row" on ML-25M.
  * QBandit's latent factor model is **PureSVD**.

===============================================================================================
THE ABLATION -- PRE-REGISTERED BEFORE RUNNING (this docstring is the pre-registration)
===============================================================================================
S6.5 states plainly:

    "The NDCG of Q_Bandit converges or even decreases after ten questions.  **We think that the
     main reason for the decreasing NDCG is that the questionnaires are not allowed to recommend
     items that they contain as questions.**  Therefore, using popular items as questions may
     have a negative impact on the recommendation performance if the questionnaire is unable to
     generalize the gained information."

★ STRUCTURAL POINT THAT DECIDES THE DESIGN.  Their candidate rule is already "all items with
x_ui = 0", i.e. every item the questionnaire ANSWERED is out of the pool anyway.  So the extra
constraint I_Q ∩ I_R = ∅ bites on exactly one set: **asked-but-UNANSWERED items**.  And under
their offline protocol an asked-but-unanswered item is one the test user has **no rating for**,
hence a *guaranteed non-target* (gain 0).  Removing a guaranteed non-target from I_R cannot
lower DCG and cannot change IDCG.

    ==> The rule S6.5 blames is, by construction, a MONOTONE BENEFIT, not a tax.
    ==> The tax must live in the OTHER exclusion -- the answered one, which their design forces
        because a test user has no held-out split: every question the user CAN answer deletes a
        real, high-gain, easy target from that user's own ground truth.  A questionnaire that
        asks popular items answers more often, so it eats more of its own ground truth, and the
        cost compounds with k.  That is a different mechanism from the one they name.

THREE EVALUATION ARMS, everything else identical (same interviews where the answers are
identical, same recommender, same users, same k):

  ARM C  "theirs"          I_R = {x_ui = 0} \ I_Q          targets = user's ratings ∩ I_R
  ARM B  "exclusion off"   I_R = {x_ui = 0}                targets = user's ratings ∩ I_R
         (drops I_Q ∩ I_R = ∅ only; answered items still out, as their own x_ui=0 rule requires)
  ARM A  "no self-eating"  a fixed per-user 20% of the ratings is held out BEFORE the interview,
         is never answerable, and is the ONLY target set.  I_R = I \ (ask-side rated).
         Arm-independent ground truth; no questionnaire can consume its own targets.

PRE-REGISTERED OUTCOMES (stated before running, per the brief):
  (a) QBandit's decline vanishes under ARM B  ->  a published result is a protocol artefact of
      the rule the authors name.  **We predict this will NOT happen**, for the structural reason
      above; if it does, our reading of their candidate rule is wrong and that is itself the
      finding.
  (b) The decline persists under ARM B  ->  I_Q ∩ I_R = ∅ is not the cause and S6.5's diagnosis
      is wrong.  ARM A then adjudicates the real cause.
  (c) Partial -> quantify the share of the decline attributable to each exclusion.
  ARM A is the discriminating cell: if the decline vanishes there, the cause is
  ground-truth consumption by answered questions (their no-held-out-split design), not the
  asked-item rule.

===============================================================================================
PERSISTENCE (CLAUDE.md HARD RULE 11)
===============================================================================================
Every (questionnaire, arm, k) cell is written to the output JSON the moment it completes.
The two expensive intermediates -- the PureSVD factors and the item-item gram G = X^T X -- are
cached in contract-signed .npz sidecars keyed on a hash of (dataset, split seed, n_train_users,
rank/threshold).  Resume refuses across a changed contract.  A unique --label stamps every
output path; the script refuses to open an output another live process holds.
"""
import os
import sys
import json
import time
import hashlib
import argparse

import numpy as np
import scipy.sparse as sp

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))

RAW = os.path.join(_ROOT, "data", "movielens", "ratings.csv")
CACHE = os.path.join(_ROOT, ".cache", "greedyslim")
OUTDIR = os.path.join(_ROOT, "experiments", "baselines", "greedyslim")

# --- their published targets, for the snap verdict table --------------------------------------
PUBLISHED = {
    ("qgslim", "all"):  {5: .3390, 10: .3594, 15: .3709, 20: .3752},
    ("qbandit", "all"): {5: .3382, 10: .3456, 15: .3221, 20: .2916},
    ("qgslim", "tail"): {5: .1163, 10: .1543, 15: .1674, 20: .1779},
    ("qbandit", "tail"):{5: .1183, 10: .1465, 15: .1488, 20: .1466},
}
BUDGETS = (5, 10, 15, 20)
LAMBDA_1 = 2.0 ** 12
LAMBDA_F = 2.0 ** 16


def logln(msg, path=None):
    line = "[%s] %s" % (time.strftime("%H:%M:%S"), msg)
    print(line, flush=True)
    if path:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")


# =============================================================================================
# DATA -- ML-25M unfiltered, exactly as they use it
# =============================================================================================
def load_ml25m(log=None):
    """Full ML-25M as a CSR users x items matrix of explicit 1-5 ratings.  No filtering."""
    os.makedirs(CACHE, exist_ok=True)
    cpath = os.path.join(CACHE, "ml25m_full.npz")
    if os.path.exists(cpath):
        z = np.load(cpath)
        X = sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))
        logln("loaded cached ML-25M %s nnz=%d" % (X.shape, X.nnz), log)
        return X
    import pandas as pd
    logln("reading %s ..." % RAW, log)
    df = pd.read_csv(RAW, usecols=["userId", "movieId", "rating"],
                     dtype={"userId": np.int64, "movieId": np.int64, "rating": np.float32})
    logln("read %d ratings, %d users, %d items" % (len(df), df.userId.nunique(),
                                                   df.movieId.nunique()), log)
    uu, urow = np.unique(df.userId.values, return_inverse=True)
    ii, icol = np.unique(df.movieId.values, return_inverse=True)
    X = sp.csr_matrix((df.rating.values.astype(np.float32), (urow, icol)),
                      shape=(len(uu), len(ii)))
    X.sort_indices()
    np.savez(cpath, data=X.data, indices=X.indices, indptr=X.indptr, shape=np.array(X.shape))
    logln("built ML-25M %s nnz=%d -> %s" % (X.shape, X.nnz, cpath), log)
    return X


def split_users(n_users, seed=98765):
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n_users)
    n_test = int(round(0.1 * n_users))
    return np.sort(perm[n_test:]), np.sort(perm[:n_test])   # train, test


def head_mask(Xtr, frac=0.33):
    """Cremonesi 33%-of-ratings short head, computed on the training interactions."""
    cnt = np.asarray((Xtr > 0).sum(axis=0)).ravel().astype(np.int64)
    order = np.argsort(-cnt, kind="stable")
    cum = np.cumsum(cnt[order])
    n_head = int(np.searchsorted(cum, frac * cnt.sum()) + 1)
    head = np.zeros(Xtr.shape[1], dtype=bool)
    head[order[:n_head]] = True
    return head, n_head


# =============================================================================================
# METRIC -- their NDCG@10 exactly: gain 2^r - 1, IDCG over the candidate set I_R
# =============================================================================================
def ndcg_at_k_block(scores, gains, cand, k=10):
    """scores, gains, cand : (B, n_items).  cand is a boolean candidate mask (I_R).
    Returns (B,) NDCG@k with gain already in `gains` (0 where not a target).
    IDCG is the best achievable DCG@k using only items in I_R -- their definition."""
    B = scores.shape[0]
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    s = np.where(cand, scores, -np.inf)
    top = np.argpartition(-s, k, axis=1)[:, :k]
    rows = np.arange(B)[:, None]
    tv = s[rows, top]
    ordr = np.argsort(-tv, axis=1, kind="stable")
    top = top[rows, ordr]
    g = np.where(cand[rows, top], gains[rows, top], 0.0)
    dcg = (g * disc[None, :]).sum(axis=1)

    gc = np.where(cand, gains, 0.0)
    ktop = np.argpartition(-gc, k, axis=1)[:, :k]
    iv = np.sort(gc[rows, ktop], axis=1)[:, ::-1]
    idcg = (iv * disc[None, :]).sum(axis=1)
    out = np.zeros(B, dtype=np.float64)
    nz = idcg > 0
    out[nz] = dcg[nz] / idcg[nz]
    return out, nz


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    a = ap.parse_args()
    if a.probe:
        os.makedirs(OUTDIR, exist_ok=True)
        lg = os.path.join(OUTDIR, "probe.log")
        X = load_ml25m(lg)
        tr, te = split_users(X.shape[0])
        logln("split: train=%d test=%d" % (len(tr), len(te)), lg)
        Xtr = X[tr]
        hd, nh = head_mask(Xtr)
        logln("head items = %d of %d (33%% of train ratings)" % (nh, X.shape[1]), lg)
        pl = np.diff(X.indptr)
        logln("ratings/user: mean=%.1f median=%d max=%d" % (pl.mean(), np.median(pl), pl.max()), lg)
