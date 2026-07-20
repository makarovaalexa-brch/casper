"""arena_clone.py -- THE CLONE TEST (author top-priority diagnostic).

Can the class-A adaptive architecture (same belief state / fold-v3.1 / blind candidate features /
candidate rule) even REPRODUCE the static b2 -- a policy it strictly contains? Train the scorer by
pure IMITATION of b2's exact sequence (supervised: at each turn the target is b2's next question;
binary relevance over the candidate set, GBM classifier, argmax at deploy). No myopic labels.

VERDICT A (clone ties b2): the class is expressive enough -- the disease is the training OBJECTIVE.
VERDICT B (clone below b2): representation bottleneck -- the state/features cannot identify a fixed
positional sequence. State-encoding probe (from arena_policies._feat): the policy input encodes the
turn index (turn_frac), answered/refused COUNTS (n_ans/n_ref) and the belief z + per-candidate
(channel one-hot, blind p_ans, belief-alignment, |z|, blind value magnitude) -- it does NOT encode
question identity or the asked-set's contents; used-set exclusion lives in the candidate rule.

DEV synthetic only; $0; no LLM; the 173/300 untouched. Writes '## CLONE TEST' to ARENA_BUILD.md.
"""
import os, sys, json, time
import numpy as np
import scipy.sparse as sp

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE); sys.path.insert(0, os.path.join(_HERE, "instrument2"))
import warnings
warnings.filterwarnings("ignore")

import arena_core as AC
import arena_policies as AP
from arena_core import paired_ci, ndcg_at_k_batch, TYPE_CONCEPT, TYPE_ENTITY, TYPE_ITEM
from arena_policies import (StaticSeq, Policy, top_answerable, run_policy)

MD = "experiments/ARENA_BUILD.md"
Tmax = 24; KS = (50, 10); K = 50
N_TRAIN_USERS = 150      # imitation states from b2's construction cohort (subset for compute)
N_NEG = 30               # negatives per state from the candidate set
CAND_M = 100             # the arena's shared deploy-time candidate rule


def md(t):
    open(MD, "a", encoding="utf-8").write(t)


def endpoint(o, K_=50, t=Tmax):
    return float(np.nanmean(o["curves"][K_][:, t]))


def _fmt(c):
    return f"{c['mean']:+.4f} [{c['lo']:+.4f},{c['hi']:+.4f}] MDE {c['mde']:.4f} (n={c['n']})"


class _FeatAll:
    """Vectorised blind scorer features for all candidates at a state (same fields as _feat)."""
    def __init__(self, ar):
        rows, cols = [], []
        for qi in range(ar.nQ):
            for j in ar.region_members(qi):
                rows.append(qi); cols.append(int(j))
        self.Qm = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(ar.nQ, ar.uni.ni))
        self.Qcnt = np.asarray(self.Qm.sum(1)).ravel() + 1e-9
        self.ar = ar

    def __call__(self, z, n_ans, n_ref, t):
        ar = self.ar
        s = ar.FR.decode_np(z[None, :])[0]
        mu = float(np.median(s)); sd = float(np.std(s) + 1e-9)
        znorm = float(np.linalg.norm(z) + 1e-9)
        pans = ar.blind_pans_all(z, znorm)
        align = (ar.Qemb @ z) / (znorm * ar.Qemb_norm) if znorm > 1e-9 else np.zeros(ar.nQ)
        vmag = np.abs(np.clip((np.asarray(self.Qm.dot(s)).ravel() / self.Qcnt - mu) / (2 * sd), -1, 1))
        X = np.zeros((ar.nQ, 11))
        X[:, 0] = (ar.q_channel == TYPE_ITEM); X[:, 1] = (ar.q_channel == TYPE_CONCEPT)
        X[:, 2] = (ar.q_channel == TYPE_ENTITY)
        X[:, 3] = pans; X[:, 4] = align; X[:, 5] = znorm; X[:, 6] = vmag
        X[:, 7] = n_ans; X[:, 8] = n_ref; X[:, 9] = t / Tmax; X[:, 10] = 2.0
        return X


class CloneScorer(Policy):
    """Deploy exactly like ScorerA (same candidate rule) but scoring with the imitation classifier."""
    name = "A_clone"
    def __init__(self, clf, fa, M=CAND_M):
        self.clf = clf; self.fa = fa; self.M = M
    def pick(self, arena, i, uid, view, asked, ansf, used, tokens, natives, z_cur, t):
        n_ans = int(sum(ansf)); n_ref = int(len(ansf) - n_ans)
        znorm = float(np.linalg.norm(z_cur) + 1e-9)
        cand = top_answerable(arena, z_cur, znorm, used, self.M)
        if not cand:
            return None
        X = self.fa(z_cur, n_ans, n_ref, t)[cand]
        p = self.clf.predict_proba(X)[:, 1] if hasattr(self.clf, "predict_proba") \
            else self.clf.predict(X)
        return cand[int(np.argmax(p))]


def main():
    from sklearn.ensemble import HistGradientBoostingClassifier
    t0 = time.time()
    ar = AC.Arena()
    coh = AC.make_cohorts(ar, n_train=1000, n_devval=80, n_devtest=160)
    cfg = coh["cfg"]
    ar.prefill_answers(coh["train"], "train", cfg, verbose=False)
    ar.prefill_answers(coh["devtest"], "devtest", cfg, verbose=False)
    ar.set_pop_prior(AP.train_pop_prior(ar, coh["train"]))
    b2_seq, _ = AP.build_b2(ar, coh["train"][:300], Tmax=Tmax, prescreen_top=300, verbose=False)
    fa = _FeatAll(ar)
    rng = np.random.default_rng(AC.SEED)

    # ---- imitation dataset: states along b2's own trajectory on its construction users ----
    print(f"[clone] building imitation states on {N_TRAIN_USERS} b2-construction users ...", flush=True)
    X_all = []; y_all = []
    reach = 0; tot_states = 0
    for n_u, rec in enumerate(coh["train"][:N_TRAIN_USERS]):
        uid = rec["u"]; ctx = ar.user_ctx(rec)
        toks = []; nat = []; used = set()
        n_ans = 0; n_ref = 0
        for t in range(Tmax):
            z = ar.belief_z_batch([toks], [nat])[0]
            q_star = int(b2_seq[t])
            znorm = float(np.linalg.norm(z) + 1e-9)
            cand = top_answerable(ar, z, znorm, used, CAND_M)
            tot_states += 1
            if q_star in cand:
                reach += 1
            X = fa(z, n_ans, n_ref, t)
            negs = [q for q in cand if q != q_star]
            if len(negs) > N_NEG:
                negs = list(rng.choice(negs, size=N_NEG, replace=False))
            X_all.append(X[q_star]); y_all.append(1)
            for q in negs:
                X_all.append(X[q]); y_all.append(0)
            # advance the b2 trajectory
            used.add(q_star)
            toks += ar.tokens_for(uid, q_star, ctx)
            if ar.answered(uid, q_star, ctx) and ar.is_liked_item(uid, q_star, ctx):
                nat.append(int(ar.uni.bank[q_star - ar.off_item]))
            if ar.answered(uid, q_star, ctx):
                n_ans += 1
            else:
                n_ref += 1
        if (n_u + 1) % 30 == 0:
            print(f"    [clone] {n_u+1}/{N_TRAIN_USERS} users [{time.time()-t0:.0f}s]", flush=True)
    X_all = np.array(X_all); y_all = np.array(y_all)
    reach_rate = reach / max(tot_states, 1)
    print(f"[clone] {len(X_all)} rows ({int(y_all.sum())} positives); b2-pick REACHABILITY inside "
          f"the cand_M={CAND_M} rule: {reach_rate:.3f}", flush=True)
    clf = HistGradientBoostingClassifier(max_iter=400, max_depth=6, learning_rate=0.08,
                                         min_samples_leaf=30, random_state=AC.SEED)
    clf.fit(X_all, y_all)
    acc = float(clf.score(X_all, y_all))
    import joblib
    joblib.dump(clf, f"{AC.CACHE_DIR}/clone_clf.joblib")
    print(f"[clone] classifier trained (train acc {acc:.3f})", flush=True)

    # ---- evaluate on IN-SAMPLE (train[:200]) and DEV-TEST (160), paired vs b2 ----
    results = {}
    for tag, cohu in (("in-sample", coh["train"][:200]), ("DEV-TEST", coh["devtest"])):
        o_c = run_policy(ar, cohu, CloneScorer(clf, fa), Tmax, Ks=KS, tag=f"clone@{tag}")
        o_b = run_policy(ar, cohu, StaticSeq("b2", b2_seq), Tmax, Ks=KS, tag=f"b2@{tag}")
        d = paired_ci([x - y for x, y in zip(o_c["curves"][50][:, Tmax], o_b["curves"][50][:, Tmax])])
        # order-match: fraction of turns where the clone asked exactly b2's t-th question
        match = []
        for i in range(len(cohu)):
            m = [1.0 if o_c["asked"][i][t] == o_b["asked"][i][t] else 0.0
                 for t in range(min(len(o_c["asked"][i]), len(o_b["asked"][i])))]
            match.append(np.mean(m) if m else 0.0)
        div = []
        for i in range(len(cohu)):
            dvt = Tmax
            for t in range(min(len(o_c["asked"][i]), Tmax)):
                if o_c["asked"][i][t] != o_b["asked"][i][t]:
                    dvt = t + 1; break
            div.append(dvt)
        results[tag] = dict(clone=endpoint(o_c), b2=endpoint(o_b), delta=d,
                            order_match=float(np.mean(match)), div_turn=float(np.mean(div)))
        print(f"  [clone {tag}] clone {endpoint(o_c):.4f} vs b2 {endpoint(o_b):.4f}  "
              f"delta {_fmt(d)}  order-match {np.mean(match):.2f}  first-div turn "
              f"{np.mean(div):.1f}", flush=True)

    # ---- verdict + report ----
    din = results["in-sample"]["delta"]; ddev = results["DEV-TEST"]["delta"]
    tie_in = din["lo"] <= 0 <= din["hi"] or din["lo"] > 0
    tie_dev = ddev["lo"] <= 0 <= ddev["hi"] or ddev["lo"] > 0
    verdict_A = tie_in and tie_dev
    md("\n\n---\n\n## CLONE TEST (author top-priority: can the architecture reproduce the static?)\n\n")
    md("The class-A architecture (same belief state, fold-v3.1, blind candidate features, "
       f"cand_M={CAND_M} rule) trained by PURE IMITATION of b2's exact sequence (binary relevance, "
       f"GBM classifier, argmax; {len(X_all)} rows from {N_TRAIN_USERS} construction users; train "
       f"acc {acc:.3f}). b2-pick reachability inside the candidate rule: {reach_rate:.3f} "
       "(if <1, the candidate cap itself blocks perfect cloning).\n\n")
    md("| cohort | clone @50 T24 | b2 @50 T24 | clone - b2 (paired) | order-match | first-divergence "
       "turn |\n|---|--:|--:|---|--:|--:|\n")
    for tag in ("in-sample", "DEV-TEST"):
        r = results[tag]
        md(f"| {tag} | {r['clone']:.4f} | {r['b2']:.4f} | {_fmt(r['delta'])} | "
           f"{r['order_match']:.2f} | {r['div_turn']:.1f} |\n")
    md("\n**VERDICT: ")
    if verdict_A:
        md("A -- the clone TIES (or beats) b2 on both cohorts: the class IS expressive enough to "
           "carry the static; the in-sample losses of the myopic-label scorer are an OBJECTIVE "
           "disease (1-/2-step labels), curable with sequence-level training.**\n")
    else:
        md("B -- the clone CANNOT match b2: a REPRESENTATION bottleneck, not (only) the objective. "
           "State-encoding probe (arena_policies._feat): the policy input encodes turn index "
           "(turn_frac), answered/refused COUNTS (n_ans/n_ref) and the belief z, plus per-candidate "
           "(channel, blind p_ans, alignment, |z|, blind value magnitude) -- it does NOT encode "
           "question IDENTITY or the asked-set's contents; distinct questions with similar "
           "aggregate features are indistinguishable, so a fixed positional sequence cannot be "
           "read out reliably.**\n")
    md(f"\nMechanism tells: order-match {results['in-sample']['order_match']:.2f} in-sample / "
       f"{results['DEV-TEST']['order_match']:.2f} DEV; first divergence turn "
       f"{results['in-sample']['div_turn']:.1f} / {results['DEV-TEST']['div_turn']:.1f}; the "
       "used-set rule forbids repeats (none possible); reachability above.\n")
    json.dump(dict(reach_rate=reach_rate, acc=acc,
                   results={k: dict(clone=v["clone"], b2=v["b2"], delta=v["delta"],
                                    order_match=v["order_match"], div_turn=v["div_turn"])
                            for k, v in results.items()},
                   verdict="A" if verdict_A else "B"),
              open(f"{AC.CACHE_DIR}/clone_test.json", "w"), indent=1, default=float)
    print(f"[clone] DONE [{(time.time()-t0)/60:.1f}m] verdict={'A' if verdict_A else 'B'}", flush=True)


if __name__ == "__main__":
    main()
