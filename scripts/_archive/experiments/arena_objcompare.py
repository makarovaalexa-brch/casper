"""arena_objcompare.py -- OBJECTIVE COMPARISON: endpoint vs belief (coordinator addendum 2026-07-10).

TEST THE AUTHOR'S HYPOTHESIS: is the adaptive policy failing because the NDCG training signal is too
noisy? Train the SAME class-A scorer architecture (identical 11 blind features, identical HistGBM,
identical argmax deploy = arena_policies.ScorerA) under TWO training objectives; ONLY the label
differs:

  ARM 1 -- ENDPOINT objective (high-variance discrete): label = marginal effect of taking candidate
    q on the endpoint NDCG@10 at the HEADLINE budget T=8, via a b2-completion rollout:
      endpoint(prefix + q + b2-completion) - endpoint(prefix + b2-completion).
    Single discrete NDCG@10 rollout attributed to one action = the noisy target.

  ARM 2 -- BELIEF objective (dense low-variance, Paper B style): label = marginal belief-embedding
    gain toward the user's full-profile target z*:
      cos(z_after_q, z*) - cos(z_before, z*),  z* = fold of ALL the user's known info (privileged
    TRAINING teacher only; the deployed policy uses beliefs alone -> deployable).

Both policies are then deployed identically (free argmax over the shared candidate set) and evaluated
per-turn NDCG@10 (turns 0..25, headline T=8) on the SAME DEV-TEST cohort, vs b2 and cold references.

DECISIVE READ: does ARM 2 produce a RISING per-turn NDCG@10 where ARM 1 stays FLAT near cold?
  ARM2 rises & ARM1 flat -> hypothesis holds (NDCG signal too noisy; belief-matching is the fix).
  both flat -> deeper problem (model/representation), not the objective.

Paper B lineage: scripts/paper2/ CASPER-R / encoder_*.py train on belief/embedding reconstruction;
that code targets a different (SBERT/two-tower) stack, so the belief objective is implemented here
directly as cosine-gain to z* (stated per the coordinator's fallback). DEV/synthetic, $0, NO LLM;
the 173/300 real-judged users are untouched. Deterministic (seed 123).
"""
import os, sys, json, time
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE); sys.path.insert(0, os.path.join(_HERE, "instrument2"))
import warnings
warnings.filterwarnings("ignore")

import arena_core as AC
import arena_policies as AP
from arena_core import ndcg_at_k, ndcg_at_k_batch, paired_ci
from arena_policies import (StaticSeq, ScorerA, run_policy, top_answerable, blind_scores, _feat)
# note: blind answerability is an Arena method -> ar.blind_pans_all(...)

MD = f"{AC.CACHE_DIR}/objcompare_section.md"
CACHE = AC.CACHE_DIR
Tmax = 25
H = 8                       # headline endpoint budget
K = 10
KS = (10, 50)
N_DEVTEST = 600
N_TRAIN_USE = 400          # TRAIN users for z* teacher + state sampling
N_SAMPLES = 3000           # (state, candidate) training samples per arm (shared states)
CAND_POOL = 80


def md(t):
    open(MD, "a", encoding="utf-8").write(t)


if os.path.exists(MD):
    os.remove(MD)


def cos(a, b):
    na = np.linalg.norm(a) + 1e-9; nb = np.linalg.norm(b) + 1e-9
    return float(np.dot(a, b) / (na * nb))


def full_profile_z(ar, recs):
    """z* teacher: fold ALL universe questions (all the user's known info). Chunked."""
    parts = []
    for s in range(0, len(recs), 40):
        chunk = recs[s:s + 40]
        tl = []; nl = []
        for r in chunk:
            uid = r["u"]; ctx = ar.user_ctx(r)
            toks = []; nat = []
            for qi in range(ar.nQ):
                toks += ar.tokens_for(uid, qi, ctx)
                if ar.is_liked_item(uid, qi, ctx) and ar.answered(uid, qi, ctx):
                    nat.append(int(ar.uni.bank[qi - ar.off_item]))
            tl.append(toks); nl.append(nat)
        parts.append(np.asarray(ar.belief_z_batch(tl, nl)))
    return np.vstack(parts)


def rollout_endpoint(ar, uid, ctx, held, prof, order_q, b2_seq, horizon=H):
    """endpoint NDCG@10 at T=horizon of order_q then b2-completion (one fold)."""
    used = set(order_q); order = list(order_q)
    for qq in b2_seq:
        if len(order) >= horizon:
            break
        if qq not in used:
            order.append(qq); used.add(qq)
    toks = []; nat = []
    for qq in order[:horizon]:
        toks += ar.tokens_for(uid, qq, ctx)
        if ar.answered(uid, qq, ctx) and ar.is_liked_item(uid, qq, ctx):
            nat.append(int(ar.uni.bank[qq - ar.off_item]))
    z = ar.belief_z_batch([toks], [nat])[0]
    v = ndcg_at_k(ar.FR, z, held, prof, K)
    return 0.0 if v is None else v


def build_labels(ar, train_recs, zstar, b2_seq):
    """Sample shared states; per sample compute features + BOTH labels (endpoint-gain, belief-gain)."""
    rng = np.random.default_rng(AC.SEED)
    ctxs = {r["u"]: ar.user_ctx(r) for r in train_recs}
    X = []; Y_end = []; Y_bel = []
    zmap = {r["u"]: zstar[i] for i, r in enumerate(train_recs)}
    t0 = time.time()
    for si in range(N_SAMPLES):
        r = train_recs[rng.integers(len(train_recs))]; uid = r["u"]; ctx = ctxs[uid]
        held = r["held"]; prof = set(r["known"].keys())
        plen = int(rng.integers(0, H))                       # prefix length 0..H-1
        allq = rng.permutation(ar.nQ); used = set(); prefix = []
        n_ans = 0; n_ref = 0; pi = 0
        while len(used) < plen and pi < len(allq):
            qi = int(allq[pi]); pi += 1
            used.add(qi); prefix.append(qi)
            if ar.answered(uid, qi, ctx):
                n_ans += 1
            else:
                n_ref += 1
        toks = []; nat = []
        for qi in prefix:
            toks += ar.tokens_for(uid, qi, ctx)
            if ar.answered(uid, qi, ctx) and ar.is_liked_item(uid, qi, ctx):
                nat.append(int(ar.uni.bank[qi - ar.off_item]))
        z_before = ar.belief_z_batch([toks], [nat])[0]
        s, mu, sd = blind_scores(ar, z_before); znorm = float(np.linalg.norm(z_before) + 1e-9)
        pans = ar.blind_pans_all(z_before, znorm)
        cpool = top_answerable(ar, z_before, znorm, used, CAND_POOL)
        if not cpool:
            continue
        qi = int(cpool[rng.integers(len(cpool))])
        # features (same builder/constant horizon as ScorerA deploy)
        X.append(_feat(ar, qi, z_before, s, mu, sd, znorm, pans, n_ans, n_ref, len(used), Tmax, 2))
        # z_after for belief label
        tk = ar.tokens_for(uid, qi, ctx)
        nv = nat + ([int(ar.uni.bank[qi - ar.off_item])]
                    if ar.answered(uid, qi, ctx) and ar.is_liked_item(uid, qi, ctx) else [])
        z_after = ar.belief_z_batch([toks + tk], [nv])[0]
        zs = zmap[uid]
        Y_bel.append(cos(z_after, zs) - cos(z_before, zs))
        # endpoint marginal effect (discrete NDCG@10 rollout)
        base_end = rollout_endpoint(ar, uid, ctx, held, prof, prefix, b2_seq)
        q_end = rollout_endpoint(ar, uid, ctx, held, prof, prefix + [qi], b2_seq)
        Y_end.append(q_end - base_end)
        if (si + 1) % 500 == 0:
            print(f"    [labels] {si+1}/{N_SAMPLES} [{time.time()-t0:.0f}s]", flush=True)
    return np.array(X), np.array(Y_end), np.array(Y_bel)


def train_gbm(X, Y):
    from sklearn.ensemble import HistGradientBoostingRegressor
    g = HistGradientBoostingRegressor(max_iter=300, max_depth=4, learning_rate=0.06,
                                      min_samples_leaf=40, random_state=AC.SEED)
    g.fit(X, Y)
    return g


def curve(o, K_=10):
    return np.nanmean(o["curves"][K_], axis=0)


def main():
    import joblib
    t0 = time.time()
    ar = AC.Arena()
    coh = AC.make_cohorts(ar, n_train=1000, n_devval=80, n_devtest=N_DEVTEST)
    cfg = coh["cfg"]
    ar.prefill_answers(coh["train"], "train", cfg, verbose=False)
    ar.prefill_answers(coh["devtest"], "devtest_big", cfg, verbose=False)
    ar.set_pop_prior(AP.train_pop_prior(ar, coh["train"]))
    dt = coh["devtest"]
    train_use = coh["train"][:N_TRAIN_USE]

    b2_seq = json.load(open(f"{CACHE}/b2v31_T25_K10_n300.json"))["seq"]
    b2_seq = [int(q) for q in b2_seq][:Tmax]

    print(f"[obj] z* teacher: full-profile fold for {len(train_use)} TRAIN users ...", flush=True)
    zstar = full_profile_z(ar, train_use)

    print(f"[obj] building shared training states + dual labels ({N_SAMPLES} samples) ...", flush=True)
    X, Y_end, Y_bel = build_labels(ar, train_use, zstar, b2_seq)
    print(f"[obj] {len(X)} samples; endpoint label mean {Y_end.mean():+.4f} sd {Y_end.std():.4f} | "
          f"belief label mean {Y_bel.mean():+.4f} sd {Y_bel.std():.4f}", flush=True)

    g_end = train_gbm(X, Y_end); joblib.dump(g_end, f"{CACHE}/obj_endpoint.joblib")
    g_bel = train_gbm(X, Y_bel); joblib.dump(g_bel, f"{CACHE}/obj_belief.joblib")
    print(f"[obj] trained both GBMs (train R^2 end {g_end.score(X,Y_end):.3f} / "
          f"bel {g_bel.score(X,Y_bel):.3f})", flush=True)

    print(f"[obj] evaluating per-turn NDCG@10 on {len(dt)} DEV-TEST users ...", flush=True)
    o_b2 = run_policy(ar, dt, StaticSeq("b2", b2_seq), Tmax, Ks=KS, tag="obj_b2")
    o_end = run_policy(ar, dt, ScorerA(g_end, M=100, Tmax=Tmax), Tmax, Ks=KS, tag="obj_end")
    o_bel = run_policy(ar, dt, ScorerA(g_bel, M=100, Tmax=Tmax), Tmax, Ks=KS, tag="obj_bel")

    c_b2 = curve(o_b2); c_end = curve(o_end); c_bel = curve(o_bel)
    cold = float(c_b2[0])

    # per-user matrices for paired CI at turn 8
    U = {"cold_col": o_b2["curves"][10][:, 0],
         "b2_8": o_b2["curves"][10][:, H], "end_8": o_end["curves"][10][:, H],
         "bel_8": o_bel["curves"][10][:, H]}
    lift = lambda a, b: paired_ci([x - y for x, y in zip(a, b)])
    end_vs_cold = lift(U["end_8"], U["cold_col"]); bel_vs_cold = lift(U["bel_8"], U["cold_col"])
    end_vs_b2 = lift(U["end_8"], U["b2_8"]); bel_vs_b2 = lift(U["bel_8"], U["b2_8"])

    def rising(ci):
        return ci["lo"] > 0
    end_rises = rising(end_vs_cold); bel_rises = rising(bel_vs_cold)
    if bel_rises and not end_rises:
        verdict = ("HYPOTHESIS HOLDS: the belief-trained policy RISES above cold while the "
                   "endpoint-trained policy stays FLAT near cold-start -- the NDCG endpoint signal "
                   "was too noisy; the dense belief-matching objective is the fix.")
    elif not bel_rises and not end_rises:
        verdict = ("HYPOTHESIS REJECTED: BOTH arms stay flat near cold-start -- the failure is "
                   "NOT the objective's variance but a deeper model/representation limit "
                   "(the free scorer cannot turn any per-question signal into ranking gain here).")
    elif bel_rises and end_rises:
        verdict = ("MIXED: BOTH arms rise above cold -- the endpoint signal is usable too; the "
                   "objective is not the sole bottleneck (compare their magnitudes below).")
    else:
        verdict = ("SURPRISE: the endpoint-trained arm rises while belief does not -- opposite of "
                   "the hypothesis; the belief teacher/deploy is mis-specified here.")

    def fmt(c):
        return f"{c['mean']:+.4f} [{c['lo']:+.4f},{c['hi']:+.4f}] MDE {c['mde']:.4f} (n={c['n']})"

    # ---- console ----
    print("\n[obj] PER-TURN NDCG@10 (DEV-TEST n=%d)" % len(dt), flush=True)
    print("turn |   b2   | endpoint | belief   (cold=%.4f)" % cold, flush=True)
    for t in range(0, Tmax + 1):
        print(f"  {t:2d} | {c_b2[t]:.4f} | {c_end[t]:.4f} | {c_bel[t]:.4f}", flush=True)
    print(f"\nturn-{H} lifts: endpoint vs cold {fmt(end_vs_cold)}; belief vs cold {fmt(bel_vs_cold)}",
          flush=True)
    print(f"turn-{H} lifts: endpoint vs b2 {fmt(end_vs_b2)}; belief vs b2 {fmt(bel_vs_b2)}", flush=True)
    print("VERDICT:", verdict, flush=True)

    # ---- persist ----
    json.dump(dict(cold=cold, b2=[float(x) for x in c_b2], endpoint=[float(x) for x in c_end],
                   belief=[float(x) for x in c_bel], H=H,
                   end_vs_cold=end_vs_cold, bel_vs_cold=bel_vs_cold,
                   end_vs_b2=end_vs_b2, bel_vs_b2=bel_vs_b2,
                   end_label_sd=float(Y_end.std()), bel_label_sd=float(Y_bel.std()),
                   verdict=verdict),
              open(f"{CACHE}/objcompare.json", "w"), indent=1, default=float)

    # ---- MD ----
    md("\n\n---\n\n## OBJECTIVE COMPARISON: endpoint vs belief\n\n")
    md("Coordinator addendum 2026-07-10 -- author hypothesis: the adaptive policy fails because the "
       "NDCG training signal is too noisy. SAME class-A scorer architecture (identical 11 blind "
       "features, identical HistGBM, identical argmax deploy = ScorerA), ONLY the training LABEL "
       "differs. SAME gated v2.1 world (sha f53e23a7692d) + EASE backbone, SAME fold-v3 (val "
       "0.4356), SAME 2,428-q universe, SAME DEV-TEST cohort (n=%d). Headline budget T=%d, curve to "
       "%d. Training: %d TRAIN users, %d shared (state,question) samples. DEV synthetic only; the "
       "173/300 real users untouched; $0; no LLM.\n\n" % (len(dt), H, Tmax, len(train_use), len(X)))
    md("- **ARM 1 ENDPOINT** (high-variance discrete): label = marginal effect of the candidate on "
       "endpoint NDCG@10 at T=%d via b2-completion rollout, endpoint(prefix+q+compl) - "
       "endpoint(prefix+compl). Label sd = %.4f.\n" % (H, float(Y_end.std())))
    md("- **ARM 2 BELIEF** (dense low-variance, Paper B style): label = cos(z_after_q, z*) - "
       "cos(z_before, z*), z* = fold of ALL the user's known info (privileged training teacher; the "
       "deployed policy is belief-only = deployable). Paper B's SBERT/two-tower reconstruction code "
       "(scripts/paper2/encoder_*.py, CASPER-R) targets a different stack, so the belief objective "
       "is implemented directly as cosine-gain to z*. Label sd = %.4f.\n\n" % float(Y_bel.std()))
    md("| turn | b2 NDCG@10 | endpoint-policy | belief-policy |\n|--:|--:|--:|--:|\n")
    md("| 0 (cold) | %.4f | %.4f | %.4f |\n" % (cold, c_end[0], c_bel[0]))
    for t in range(1, Tmax + 1):
        mk = "  <- headline" if t == H else ""
        md("| %d%s | %.4f | %.4f | %.4f |\n" % (t, mk, c_b2[t], c_end[t], c_bel[t]))
    md("\n**Turn-%d lifts (paired bootstrap CI):**\n\n" % H)
    md("| contrast | endpoint-policy | belief-policy |\n|---|---|---|\n")
    md("| vs cold (T0) | %s | %s |\n" % (fmt(end_vs_cold), fmt(bel_vs_cold)))
    md("| vs b2 (T%d) | %s | %s |\n\n" % (H, fmt(end_vs_b2), fmt(bel_vs_b2)))
    md("**VERDICT: %s**\n" % verdict)
    print(f"[obj] DONE [{(time.time()-t0)/60:.1f}m]; wrote {MD}", flush=True)


if __name__ == "__main__":
    main()
