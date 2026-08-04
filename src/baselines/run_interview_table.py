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
  eddi            EDDI's Partial-VAE (Ma et al. ICML'19) as a CONSUMER: a permutation-invariant set
                  encoder over the observed coordinates, so it eats the same binary likes CSR as
                  recvae_likes. Trained checkpoint .cache/baselines/eddi_ml25m_liang.pt (full-profile
                  0.2153/0.1077). Its NATIVE question-selection (myopic info-gain) is a SELECTOR, not
                  a consumer -- see adaptive_harness.py / eddi_selector.py.
  mostpop         the PRIOR: ignores the answers entirely, so it is ONE number, not a curve --
                  a horizontal reference line, evaluated once

WHO THE FLOOR APPLIES TO -- "no answers" is NOT "no input" (found 2026-07-30, after the floor made
Golbandi WORSE on every zero-answer arm: 0.1650 -> 0.1643 pure-entropy, 0.1658 -> 0.1655 random-bank).
Golbandi's routing is TERNARY: like / dislike / UNKNOWN. A user asked eight popular films who rates
none of them does not land at the global root -- they land in the leaf of training users who also
rated none of those eight, and "I have seen none of these" is a genuinely informative answer. Flooring
that to popularity deletes a real prediction.
So the rule is: THE FLOOR APPLIES WHEN THE MODEL RECEIVED NO INPUT, not when the user gave no
like/dislike. Models that consume unknowns (golbandi_leaf) always have input and are never floored;
models that fold only answered tokens (ours, rbmf, belief_mf, the recvae backbone) get nothing at all
when nothing is answered, and are floored. Uniform as a RULE, asymmetric in outcome -- and that
asymmetry is a real capability difference that belongs in the write-up, not smoothed away.
NOTE FOR THE PAPER: our instrument DISCARDS the unknown signal. An unanswered question contributes
literally nothing to the fold. That is a concrete design gap this experiment exposed.

THE MOST-POPULAR FLOOR (author ruling, 2026-07-30). A recommender that has been told NOTHING about a
user must not do worse than showing them popular items -- any deployed system falls back that way, and
a method that scores below the popularity prior at zero evidence is being reported with its cold-start
handling removed, not reproduced. So the floor is applied UNIFORMLY: for any user with zero answers,
every recommender returns the popularity ranking. Users with >=1 answer are on their own, which is the
honest test of answer conversion. No hyperparameter, no per-model tuning, deployment-realistic, and
applied identically to all -- the same principle as the fixed candidate pool.

Without it the table reports artefacts, not methods: RBMF returns an all-zero score vector at zero
evidence (0.0003 on the arms nobody answers) and the un-adapted RecVAE backbone collapses to 0.0099 at
k=1. Neither number is about elicitation. Methods whose papers specify no cold-start fallback get a
FOOTNOTE, not a degenerate row.

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
sys.path.insert(0, os.path.join(_ROOT, "src", "instrument"))
from i26_encoder import UNSEEN_LEVEL

OUT = os.path.join(_ROOT, "experiments", "baselines", "interview")
RECS = ["ours", "recvae_likes", "golbandi_leaf", "rbmf", "belief_mf", "tanp", "eddi"]
# Models whose input contract INCLUDES the unknowns, so they are never information-free -> never floored.
CONSUMES_UNKNOWNS = {"golbandi_leaf"}
GLOBAL_ARMS = {"popularity", "pure_entropy", "entropy0", "helf"}


def logln(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "run.log"), "a") as f:
        f.write(line + "\n")


# ------------------------------------------------------------------ answer -> per-contract input
def answers_to_csr(answered, n_users, n_items, mode, asked=None):
    """mode='levels'  -> level+1 (the tower's graded encoding)
       mode='levels_unseen' -> as 'levels', PLUS asked-but-unanswered items at UNSEEN_LEVEL+1.

    THE GAP THIS CLOSES (found 2026-07-31, author: "i would expect epoch 1 to already improve on short,
    i don't know why it wouldn't"). Every interview number produced until now fed the model ONLY the
    ANSWERED items -- `asked` was consumed by Golbandi alone. So the i26 exposure branch, the entire
    point of that arm, was never exercised at evaluation: the model was trained on a channel the harness
    does not hand it. Under HELF that is 6.4 of 8 questions silently discarded before the model sees
    them. i26 numbers measured this way cannot improve on short interviews by construction.
       mode='signed'  -> level_to_sv: +1 at 5.0 stars, 0 at 2.75, -1 at 0.5 (RBMF / belief-MF targets)
       mode='likes'   -> 1.0 for level>=7 (r>=4.0) only; dislikes are DROPPED, which is precisely what
                         a binary recommender's contract can express."""
    rows, cols, vals = [], [], []
    for u, toks in enumerate(answered):
        seen = set()
        for (i, lv) in toks:
            seen.add(int(i))
            if mode == "likes":
                if lv < 7:
                    continue
                v = 1.0
            elif mode in ("levels", "levels_unseen"):
                v = float(lv) + 1.0
            else:
                v = float(level_to_sv(lv))
            rows.append(u); cols.append(i); vals.append(v)
        if mode == "levels_unseen" and asked is not None:
            for i in asked[u]:
                if int(i) not in seen:            # asked, and the user could not answer
                    rows.append(u); cols.append(int(i)); vals.append(float(UNSEEN_LEVEL) + 1.0)
    return sparse.csr_matrix((np.asarray(vals, np.float32), (rows, cols)),
                             shape=(n_users, n_items), dtype=np.float32)


def apply_pop_floor(scores, answered, pop_scores, lo, hi):
    """Users in [lo,hi) with NO answers get the popularity ranking. Uniform across recommenders."""
    n_floored = 0
    for r in range(hi - lo):
        if len(answered[lo + r]) == 0:
            scores[r] = pop_scores
            n_floored += 1
    return n_floored


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


# ------------------------------------------------------------------ recommender construction
def build_recommenders(recs, D, arch="i25", snapshot=SNAP_DEFAULT, log=logln):
    """Build every requested consumer as (input_mode, object). EXTRACTED VERBATIM from main() on
    2026-08-04 so the adaptive harness drives the IDENTICAL consumers rather than a second copy that
    can drift. Nothing in the bodies changed; only the surrounding loop moved out."""
    ni = D["n_items"]
    _hm, cnt = compute_head_mask(D["train"], ni)
    built = {}
    if "ours" in recs:
        ma = argparse.Namespace(arch=arch, teacher="warm_init", t_hidden=600, t_latent=200,
                                token="film", train_decoder=False, sign_prior=True, unfreeze_emb=False,
                                lr=3e-4, warm_lr_scale=0.1, full_kd=False, full_kd_w=0.3)
        if arch == "i26":
            sys.path.insert(0, os.path.join(_ROOT, "src", "instrument"))
            from i26_encoder import build_i26
            from train_tower_t2 import load_recvae_teacher, apply_sign_prior
            src = load_recvae_teacher(ni, hidden=600, latent=200)
            enc, dec, _p26, _g26 = build_i26(ni, src, ma, log=log)
            apply_sign_prior(enc)
        else:
            enc, dec, _t, _p, _g = build_model(ma, ni, cnt)
        blob = torch.load(snapshot, map_location="cpu")
        if arch == "i26":
            # A certified i25 checkpoint has no exposure-branch keys, so strict=True rejects it -- but
            # loading it into i26 is exactly the BEFORE-TRAINING control we want (zero-init branch =>
            # unseen tokens inert => it must reproduce the i25 numbers). Allow it, but ALLOWLIST what
            # may be missing: blanket strict=False is how a partial model gets loaded in silence.
            ALLOWED = {"W_white", "gate_e", "psi.0.weight", "psi.0.bias", "psi.2.weight", "psi.2.bias",
                       "rho_expo.0.weight", "rho_expo.0.bias", "rho_expo.2.weight", "rho_expo.2.bias"}
            miss = enc.load_state_dict(blob["enc"], strict=False)
            unexpected = set(miss.unexpected_keys)
            bad = set(miss.missing_keys) - ALLOWED
            if bad or unexpected:
                raise SystemExit(f"[interview] REFUSING to load: unexpected={sorted(unexpected)} "
                                 f"missing-and-not-allowed={sorted(bad)}")
            if miss.missing_keys:
                log(f"[interview] BEFORE-TRAINING control: exposure branch left at zero-init "
                    f"({len(miss.missing_keys)} keys) -- unseen tokens are inert, so this must "
                    f"reproduce the certified i25 numbers")
        else:
            enc.load_state_dict(blob["enc"])
        dec.load_state_dict(blob["decoder"]); enc.eval()
        log(f"[interview] ours = {arch} from {os.path.basename(snapshot)} "
            f"(ep={blob.get('epoch','?')} val_full={blob.get('val_full','?')})")
        # i26 must RECEIVE the unseen channel it was trained on; i25 has no such channel.
        built["ours"] = ("levels_unseen" if arch == "i26" else "levels",
                         (enc, dec.weight.detach(), dec.bias.detach()))
    if "recvae_likes" in recs:
        ck = os.path.join(_ROOT, ".cache", "baselines", "recvae_ml25m_liang.pt")
        ra = recvae._defaults()
        ra.epochs = int(torch.load(ck, map_location="cpu")["state"]["epoch"]); ra.max_minutes = 0.0
        built["recvae_likes"] = ("likes", recvae.fit(D["train"], ni, args=ra, ckpt=ck, log=log))
    if "rbmf" in recs:
        built["rbmf"] = ("signed", rbmf_seed.fit(D["train"], ni, log=log))
    if "belief_mf" in recs:
        built["belief_mf"] = ("signed", belief_mf.fit(D["train"], ni, log=log))
    if "tanp" in recs:
        # TaNP needs no bespoke adapter: its predict takes a likes CSR, the same contract as
        # recvae_likes, so the interview harness can drive it unchanged. Epochs are pinned to the
        # trained checkpoint so this reloads rather than retrains.
        import tanp_bestefffort as tanp
        tck = os.path.join(_ROOT, ".cache", "baselines", "tanp_ml25m_liang.pt")
        ta = tanp._defaults()
        ta.epochs = int(torch.load(tck, map_location="cpu")["state"]["epoch"])
        ta.max_minutes = 0.0
        built["tanp"] = ("likes", tanp.fit(D["train"], ni, args=ta, ckpt=tck, log=log))
    if "eddi" in recs:
        # EDDI's Partial-VAE consumes the OBSERVED SET, which under the binary implicit contract is
        # exactly the likes CSR every other set-input baseline eats. No adapter, no retrain: the
        # checkpoint (trained 2026-08-03) is loaded at its best epoch.
        import eddi_pvae
        eck = os.path.join(_ROOT, ".cache", "baselines", "eddi_ml25m_liang.pt")
        built["eddi"] = ("likes", eddi_pvae.load_trained(eck, ni, log=log))
    if "golbandi_leaf" in recs:
        built["golbandi_leaf"] = ("raw", GolbandiLeaf(D["g_train"], ni, lam=8.0, log=log))
    return built


# ------------------------------------------------------------------ one (consumer, ask-set) cell
def score_arm(name, mode, obj, asked, answered, D, pop_row, no_floor=False, global_asked=None,
              batch_size=500):
    """Score ONE cell: a consumer against ONE realised (asked, answered) interview. EXTRACTED VERBATIM
    from main()'s inner loop on 2026-08-04.

    The extraction is the point: the adaptive harness changes only HOW `asked`/`answered` are produced.
    The answer->input contract, the Most-Popular floor rule, the evaluate() cursor discipline and the
    fixed arm-N candidate pool must be the SAME CODE for the static and the sequential path, because
    the regression assert in adaptive_harness only covers the ASK side. Two copies of this block would
    drift silently and the assert would never see it.

    `global_asked` is the shared ask-list when EVERY user was asked the same items (all four static
    global arms, and turn-0 of any adaptive selector); it lets GolbandiLeaf build the partition once.
    Pass None whenever the ask-lists differ per user -- exactly today's rule."""
    n, ni = D["te_tr"].shape[0], D["n_items"]
    n_ans = float(np.mean([len(t) for t in answered]))
    if mode == "raw":
        sc = obj.scores(asked, answered, global_asked=global_asked)
        base = cursor_predict(lambda s, e, _sc=sc: _sc[s:e].copy())
        Xin = D["te_tr"]
    else:
        Xi = answers_to_csr(answered, n, ni, mode, asked=asked)
        if name == "ours":
            e_, W_, b_ = obj
            base = make_graded_predict_fn(e_, W_, b_, Xi, check_nnz=False)
        else:
            base = obj
        Xin = Xi
    floored = {"n": 0, "cursor": 0}
    use_floor = (name not in CONSUMES_UNKNOWNS) and not no_floor

    def pr(X, _b=base, _st=floored, _f=use_floor):
        out = np.array(_b(X), dtype=np.float32, copy=True)
        lo = _st["cursor"]; hi = lo + X.shape[0]
        if _f:
            _st["n"] += apply_pop_floor(out, answered, pop_row, lo, hi)
        _st["cursor"] = hi
        return out
    if hasattr(base, "reset"):
        base.reset()
    res = M.evaluate(pr, Xin, D["te_te"], batch_size=batch_size, head_mask=D["head_mask"],
                     mask_X=D["pool"])
    return {"full": res["ndcg@10"], "tail": res["tail_ndcg@10"],
            "answered": n_ans, "floored": floored["n"]}


def mostpop_row(D, log=logln):
    """The popularity prior: the floor vector itself, plus its one-number NDCG row."""
    pr_pop = pop.fit(D["train"], D["n_items"], log=log)
    pop_row = np.asarray(pr_pop(D["te_tr"][:1]), dtype=np.float32).ravel().copy()
    r = M.evaluate(pr_pop, D["te_tr"], D["te_te"], batch_size=500, head_mask=D["head_mask"],
                   mask_X=D["pool"])
    log(f"[interview] mostpop PRIOR (constant, answer-independent): "
        f"full={r['ndcg@10']:.4f} tail={r['tail_ndcg@10']:.4f}")
    return pop_row, {"full": r["ndcg@10"], "tail": r["tail_ndcg@10"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budgets", default="1,2,4,8,16")
    ap.add_argument("--only", default=None)
    ap.add_argument("--arms", default=None)
    ap.add_argument("--snapshot", default=SNAP_DEFAULT)
    ap.add_argument("--arch", default="i25", choices=["i25", "i26"],
                    help="i26 = the interview-native arm (exposure branch + trainable prior latent). "
                         "Its checkpoint cannot be loaded into an i25 model: the state dict carries "
                         "psi/rho_expo/gate_e/z0 that i25 has no slots for.")
    ap.add_argument("--label", default=None, help="row name in the output JSON (default: arch)")
    ap.add_argument("--no_floor", action="store_true",
                    help="Disable the Most-Popular floor for THIS run (the A/B arm). The floor rule is "
                         "'apply when the model received NO INPUT' -- but apply_pop_floor triggers on "
                         "zero ANSWERS, which is not the same thing once the model has an unknown "
                         "channel. i26 is fed asked-but-unseen items as UNSEEN_LEVEL tokens, so a user "
                         "who answered nothing still supplies real, set-dependent evidence (measured "
                         "2026-07-31: up to 1.44 logits away from the empty set, 1.29 between two "
                         "different unseen sets). Under the floor that evidence is computed and then "
                         "overwritten by the popularity row for up to 7,843/10,000 users at k=1. Run "
                         "both arms and report the difference rather than switching silently.")
    a = ap.parse_args()
    budgets = [int(x) for x in a.budgets.split(",")]
    recs = RECS if not a.only else [r for r in RECS if r in set(a.only.split(","))]
    arms = ST.ARMS if not a.arms else [s for s in ST.ARMS if s in set(a.arms.split(","))]

    D = load_arm_n(log=logln)
    ni, n = D["n_items"], D["te_tr"].shape[0]

    glob, bank = ST.build_orders(D["g_train"], D["train"], ni, log=logln)
    lvl = ST.level_lookup(D["g_te_tr"])
    logln(f"[interview] {len(recs)} recommenders x {len(arms)} strategies x {len(budgets)} budgets")

    built = build_recommenders(recs, D, arch=a.arch, snapshot=a.snapshot, log=logln)

    results = {}
    lbl = a.label or a.arch
    outp = os.path.join(OUT, f"interview_table_{lbl}.json")

    # ---------------------------------------------------------------- MostPop: one number
    pop_row, prior = mostpop_row(D, log=logln)
    results["mostpop"] = {"_prior": prior}

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
                ga = asked[0] if arm in GLOBAL_ARMS else None
                cell = score_arm(name, mode, obj, asked, answered, D, pop_row,
                                 no_floor=a.no_floor, global_asked=ga)
                results[name][arm][f"k{k}"] = cell
                logln(f"[interview] {name:14s} {arm:14s} k={k:2d} "
                      f"answered={cell['answered']:4.1f}/{k} floored={cell['floored']:5d} "
                      f"full={cell['full']:.4f} tail={cell['tail']:.4f} "
                      f"({(time.time() - ts) / 60:.1f}m)")
                json.dump({"lit_rank": ST.LIT_RANK, "cites": ST.CITES, "budgets": budgets,
                           "arch": a.arch, "snapshot": os.path.basename(a.snapshot),
                           "results": results}, open(outp, "w"), indent=2)
    logln(f"[interview] done -> {outp}")



if __name__ == "__main__":
    main()
