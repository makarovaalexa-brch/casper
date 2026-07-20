"""arena_adaptive_eig.py -- ADAPTIVE expected-information-gain policy vs the STATIC expected-gain
order, on the SAME test subsample. DEV/synthetic ONLY; the 173/300 LLM-judged users are NEVER
touched. NO LLM calls. $0. Deterministic (seed 123).

QUESTION (design sheet spirit): the STATIC info-gain baseline ranks questions ONCE by cold expected
NDCG@10 gain on a TRAIN subsample, then asks them in a FIXED order for everybody. Does RE-RANKING
per user per turn -- using the user's CURRENT belief z_cur -- beat that fixed order?

--------------------------------------------------------------------------------------------------
AdaptiveEIG (peek-free adaptive expected-gain policy)
--------------------------------------------------------------------------------------------------
At each turn t, for the current user with belief z_cur (the fold of answers so far), and for each
UNASKED candidate q in a shortlist:

  expected_gain(q) = SUM over outcomes o in {refuse, dislike, like} of
                        P(o | z_cur, q)  *  [ NDCG@10(fold(z_cur, o_q), R_hat) - NDCG@10(z_cur, R_hat) ]

  pick argmax_q expected_gain(q), ask it, fold the REAL answer, repeat.

NO PEEK (the whole point). The gain is an EXPECTATION over MODEL-PREDICTED outcomes and is scored
against a MODEL-PREDICTED relevance target R_hat -- the user's TRUE held-out target is NEVER read
inside pick(). Concretely, everything the gain touches is derived from z_cur + the shared blind
answerability prior + blind value forecasts:

  * P(o|z_cur,q): P(refuse)=1-p_ans_blind(q); within answered, split like/dislike by the belief's
    blind forecast value v_hat(q) (decode(z_cur) over q's region members, centered/scaled):
    P(like|ans)=clip(0.5+0.5*v_hat, .05, .95). p_ans_blind = arena.blind_pans_all (belief-aligned
    population prior). Identical machinery to b4/A/B/C.
  * outcome tokens o_q: the SAME blind hypothetical tokens the audited blind arms use
    (arena_policies.hypo_tokens style): implicit know_well + explicit EASE forecast value
    (+|v_hat| for like, -|v_hat| for dislike); refuse = a blind no-clue token (surp/anti=0).
  * R_hat (peek-free relevance target) = the ANTICIPATED FULLY-ELICITED BELIEF: fold z_cur PLUS the
    single most-likely predicted answers to the top-M_hat blind-answerable unasked questions, then
    take the top-K_hat catalog items of decode(z_hat). R_hat is what the recommender WOULD believe
    after a full interview; the adaptive policy greedily asks the question whose predicted answer
    moves z_cur fastest toward that informed belief. R_hat uses ZERO held-out / true-answer info.

This is genuinely ADAPTIVE: z_cur (hence v_hat, p_ans, R_hat, and the whole ranking) changes with
every answer, unlike the static fixed order. It is the ANSWER-DISTRIBUTION-marginalized cousin of
b4 (b4 folds ONE expected answer and scores with the smooth-J surrogate; AdaptiveEIG marginalizes
over the {refuse,dislike,like} distribution and scores with the NDCG@10 machinery -- the SAME
ndcg_at_k_batch that prescreen_gains uses, for a fair static-vs-adaptive comparison).

TRACTABILITY CHOICES (flagged):
  * TEST SUBSAMPLE = first 500 devtest users of the seed-123 split (same split as baselines_v2;
    devtest = users [n_train_sub-independent] index 0..499). Static info-gain is re-run on the SAME
    500 users for an apples-to-apples delta.
  * Per turn, candidate set = top-M (default 150) UNASKED questions by current-belief blind
    answerability (top_answerable) -- a shortlist, NOT the full 2,428 (a full per-turn per-user
    2,428-scan would be ~500x the static prescreen cost). Documented deviation.
  * R_hat built from M_hat=60 predicted answers (one extra fold per user-turn), K_hat=20 target
    items. Marginalization over 3 representative outcomes (refuse/dislike/like), not the full
    4-level value grid.
  * Static info-gain order reused from the cached prescreen (pres31_baselines_v2_n800_K10.json),
    identical to baselines_v2's INFO-GAIN arm.
"""
import os, sys, json, time, argparse
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE); sys.path.insert(0, os.path.join(_HERE, "instrument2"))
import warnings
warnings.filterwarnings("ignore")

import arena_core as AC
import arena_policies as AP
from arena_core import ndcg_at_k_batch
from arena_policies import (StaticSeq, B0Cold, Policy, run_policy, train_pop_prior,
                            prescreen_gains, blind_scores, blind_value, top_answerable)
from arena_core import KIND_IMPL, KIND_EXPL, LVL_ROUGH, LVL_KW, LVL_NEG, FID_DATA, FID_EASE

MD = "experiments/ARENA_BUILD.md"
Tmax = 10
K = 10


# ============================================================ AdaptiveEIG (peek-free)
class AdaptiveEIG(Policy):
    """Per-turn argmax of the answer-distribution-marginalized expected NDCG@10 gain toward the
    anticipated fully-elicited belief R_hat. Fully BLIND (never reads held / true answers)."""
    name = "adaptive_eig"

    def __init__(self, M=150, M_hat=60, K_hat=20, vfloor=0.30):
        self.M = M; self.M_hat = M_hat; self.K_hat = K_hat; self.vfloor = vfloor

    # ---- blind hypothetical outcome tokens (same shape as arena_policies.hypo_tokens) ----
    def _ans_tokens(self, arena, qi, base_tokens, vsign, vmag):
        ch = int(arena.q_channel[qi]); emb = arena.Qemb[qi]
        return base_tokens + [(ch, KIND_IMPL, LVL_KW, 0.0, FID_DATA, 0.0, emb, 0.0),
                              (ch, KIND_EXPL, LVL_ROUGH, 0.0, FID_EASE, float(vsign * vmag), emb, 0.0)]

    def _refuse_tokens(self, arena, qi, base_tokens):
        ch = int(arena.q_channel[qi]); emb = arena.Qemb[qi]
        # blind no-clue token (v3.1 folds no-clue); surp/anti unknown blindly -> 0
        return base_tokens + [(ch, KIND_IMPL, LVL_NEG, 0.0, FID_DATA, 0.0, emb, 0.0)]

    def _rhat(self, arena, z_cur, s, mu, sd, znorm, used, natives):
        """Anticipated fully-elicited belief target: fold the single most-likely predicted answers
        to the top-M_hat blind-answerable UNASKED questions; return top-K_hat catalog items."""
        cand = top_answerable(arena, z_cur, znorm, used, self.M_hat)
        toks = []
        for qi in cand:
            v = blind_value(arena, s, mu, sd, qi)          # blind forecast in [-1,1]
            vmag = max(abs(v), self.vfloor)
            vsign = 1.0 if v >= 0 else -1.0
            # append predicted-answer tokens (implicit + explicit forecast) to the running hypo prof
            toks += self._ans_tokens(arena, qi, [], vsign, vmag)
        z_hat = arena.belief_z_batch([toks], [list(natives)])[0]
        shat = arena.FR.decode_np(z_hat[None, :])[0]
        top = np.argpartition(-shat, self.K_hat - 1)[:self.K_hat]
        return set(int(t) for t in top)

    def pick(self, arena, i, uid, view, asked, ansf, used, tokens, natives, z_cur, t):
        s, mu, sd = blind_scores(arena, z_cur)
        znorm = float(np.linalg.norm(z_cur) + 1e-9)
        cand = top_answerable(arena, z_cur, znorm, used, self.M)
        if not cand:
            return None
        pans = arena.blind_pans_all(z_cur, znorm)          # blind P(answerable) per q
        # ---- peek-free relevance target (anticipated informed belief) ----
        Rhat = self._rhat(arena, z_cur, s, mu, sd, znorm, used, natives)
        # base NDCG of the CURRENT belief toward R_hat (no profile mask; peek-free proxy)
        base = ndcg_at_k_batch(arena.FR, z_cur[None, :], [Rhat], [set()], K)[0]
        base = 0.0 if base is None else base
        # ---- build the marginalization batch: 3 outcomes per candidate ----
        tl = []; meta = []                                  # meta: (cand_idx, outcome, weight)
        for qi in cand:
            v = blind_value(arena, s, mu, sd, qi)
            vmag = max(abs(v), self.vfloor)
            p_ans = float(pans[qi])
            p_like = float(np.clip(0.5 + 0.5 * v, 0.05, 0.95))
            w_like = p_ans * p_like
            w_dis = p_ans * (1.0 - p_like)
            w_ref = 1.0 - p_ans
            tl.append(self._ans_tokens(arena, qi, list(tokens), +1.0, vmag)); meta.append((qi, w_like))
            tl.append(self._ans_tokens(arena, qi, list(tokens), -1.0, vmag)); meta.append((qi, w_dis))
            tl.append(self._refuse_tokens(arena, qi, list(tokens)));          meta.append((qi, w_ref))
        Z = arena.belief_z_batch(tl, [list(natives)] * len(tl))
        nd = ndcg_at_k_batch(arena.FR, Z, [Rhat] * len(tl), [set()] * len(tl), K)
        nd = np.array([base if v is None else v for v in nd], float)
        # marginalize per candidate
        eg = {}
        for k, (qi, w) in enumerate(meta):
            eg[qi] = eg.get(qi, 0.0) + w * (nd[k] - base)
        best_qi = max(eg, key=eg.get)
        return int(best_qi)


def curve_means(out):
    c = out["curves"][K]
    return [float(np.nanmean(c[:, t])) for t in range(Tmax + 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=14000)
    ap.add_argument("--n_devval", type=int, default=3000)
    ap.add_argument("--n_devtest", type=int, default=500)   # only need 500 test users; devtest
    # start index = n_train+n_devval is fixed, so out[17000:17500] == first 500 of the 3000-split
    # (identical users, memory-light table). cfg-sha changes -> trainsub/devtest tables regenerate
    # (deterministic per user; pop-prior + prescreen content unchanged).
    ap.add_argument("--n_train_sub", type=int, default=800)
    ap.add_argument("--n_test_sub", type=int, default=500)
    ap.add_argument("--M", type=int, default=150)
    ap.add_argument("--M_hat", type=int, default=60)
    ap.add_argument("--K_hat", type=int, default=20)
    a = ap.parse_args()
    t0 = time.time()

    ar = AC.Arena()
    coh = AC.make_cohorts(ar, n_train=a.n_train, n_devval=a.n_devval, n_devtest=a.n_devtest)
    train, dv, dt = coh["train"], coh["devval"], coh["devtest"]
    cfg = coh["cfg"]
    print(f"[split] train={len(train)} devval={len(dv)} devtest={len(dt)} (seed {AC.SEED})", flush=True)

    # ---- TRAIN subsample: priors + reuse cached info-gain prescreen (tag baselines_v2) ----
    train_sub = train[:a.n_train_sub]
    ar.prefill_answers(train_sub, "trainsub", dict(**cfg, sub=a.n_train_sub))
    ar.set_pop_prior(train_pop_prior(ar, train_sub))
    ig_gain = prescreen_gains(ar, train_sub, K=K, tag="baselines_v2")        # cached -> instant
    ig_order = [int(q) for q in np.argsort(-ig_gain)]

    # ---- TEST answers (reuse cached devtest table), take the first n_test_sub ----
    ar.prefill_answers(dt, "devtest", cfg)
    test_sub = dt[:a.n_test_sub]
    print(f"[test-sub] {len(test_sub)} users (first {a.n_test_sub} of devtest, seed {AC.SEED})",
          flush=True)

    # ---- COLD reference + STATIC info-gain on the SAME 500 users ----
    r_cold = run_policy(ar, test_sub, B0Cold(), Tmax, Ks=(K,))
    cold = curve_means(r_cold); cold0 = cold[0]
    print(f"    cold done, t0={cold0:.4f} [{time.time()-t0:.0f}s]", flush=True)
    r_ig = run_policy(ar, test_sub, StaticSeq("infogain", ig_order), Tmax, Ks=(K,))
    static_ig = curve_means(r_ig)
    print(f"    static info-gain done, t8={static_ig[8]:.4f} [{time.time()-t0:.0f}s]", flush=True)

    # ---- ADAPTIVE EIG on the SAME 500 users ----
    print("\n=== running AdaptiveEIG (peek-free) ===", flush=True)
    pol = AdaptiveEIG(M=a.M, M_hat=a.M_hat, K_hat=a.K_hat)
    r_ad = run_policy(ar, test_sub, pol, Tmax, Ks=(K,), verbose=True, tag="adaptive_eig")
    adaptive = curve_means(r_ad)
    print(f"    adaptive done, t8={adaptive[8]:.4f} [{(time.time()-t0)/60:.1f} min]", flush=True)

    # ---- deltas ----
    d8 = adaptive[8] - static_ig[8]
    lift_ad8 = adaptive[8] - cold0
    lift_ig8 = static_ig[8] - cold0
    # paired per-user bootstrap CI on the turn-8 adaptive-minus-static delta
    ca = r_ad["curves"][K][:, 8]; cs = r_ig["curves"][K][:, 8]
    paired = ca - cs
    ci = AC.paired_ci(paired)

    out = dict(
        cfg=dict(n_train=len(train), n_devval=len(dv), n_devtest=len(dt), n_train_sub=a.n_train_sub,
                 n_test_sub=len(test_sub), M=a.M, M_hat=a.M_hat, K_hat=a.K_hat, K=K, Tmax=Tmax,
                 policy="AdaptiveEIG", peek_free=True,
                 target="anticipated-informed-belief R_hat (top-K_hat of z_hat); no held-out used",
                 shortlist="top-M unasked by current-belief blind answerability",
                 marginalization="{refuse,dislike,like} weighted by blind p_ans + forecast-value split",
                 static_infogain_source="pres31_baselines_v2_n800_K10.json (same as baselines_v2)"),
        cold=cold, static_infogain=static_ig, adaptive_eig=adaptive, cold0=cold0,
        turn8=dict(adaptive=adaptive[8], static_infogain=static_ig[8],
                   delta_adaptive_minus_static=d8, lift_adaptive_over_cold=lift_ad8,
                   lift_static_over_cold=lift_ig8,
                   paired_ci=dict(mean=ci["mean"], lo=ci["lo"], hi=ci["hi"], mde=ci["mde"], n=ci["n"])),
        runtime_min=(time.time() - t0) / 60)
    os.makedirs(AC.CACHE_DIR, exist_ok=True)
    json.dump(out, open(f"{AC.CACHE_DIR}/adaptive_eig.json", "w"), indent=1)

    # ---- append section to ARENA_BUILD.md ----
    lines = []
    lines.append("\n\n## ADAPTIVE EIG vs STATIC\n\n")
    lines.append(f"Does RE-RANKING questions per user per turn (using the user's CURRENT belief "
                 f"z_cur) beat the FIXED info-gain order? Same seed-123 world (gated v2.1 + fold-v3.1 "
                 f"+ 2,428-question universe), same **{len(test_sub)}-user** TEST subsample (first "
                 f"{len(test_sub)} devtest users), NO LLM, $0. Per-turn NDCG@10.\n\n")
    lines.append("**AdaptiveEIG (peek-free):** each turn, argmax over a top-"
                 f"{a.M} blind-answerable shortlist of the answer-distribution-marginalized expected "
                 "NDCG@10 gain. Gain = SUM_o P(o|z_cur,q) * [NDCG@10(fold(z_cur,o), R_hat) - "
                 "NDCG@10(z_cur, R_hat)], o in {refuse,dislike,like} weighted by blind p_ans + "
                 "forecast-value split. **NO PEEK:** R_hat = the ANTICIPATED fully-elicited belief "
                 f"(top-{a.K_hat} items of z_hat, itself the fold of z_cur + M_hat={a.M_hat} "
                 "predicted answers); the true held-out target is NEVER read inside the gain. Same "
                 "ndcg_at_k_batch machinery as the static prescreen -> fair comparison.\n\n")
    lines.append("**STATIC info-gain:** the baselines_v2 INFO-GAIN arm -- questions ordered ONCE by "
                 "cold expected NDCG@10 gain on the 800-user TRAIN subsample, asked in a FIXED order "
                 "for everybody (reused prescreen).\n\n")
    lines.append(f"- COLD (turn 0) = **{cold0:.4f}** (reference, same 500 users).\n\n")
    lines.append("| turn | static info-gain | adaptive EIG | adaptive - static |\n")
    lines.append("|--:|--:|--:|--:|\n")
    for tt in range(1, Tmax + 1):
        tag = "  **<- headline (turn 8)**" if tt == 8 else ""
        lines.append(f"| {tt} | {static_ig[tt]:.4f} | {adaptive[tt]:.4f} | "
                     f"{adaptive[tt]-static_ig[tt]:+.4f} |{tag}\n")
    lines.append(f"\n**Turn-8:** static info-gain {static_ig[8]:.4f} (lift over cold "
                 f"{lift_ig8:+.4f}), adaptive EIG {adaptive[8]:.4f} (lift over cold {lift_ad8:+.4f}). "
                 f"**Adaptive - static = {d8:+.4f}** "
                 f"[paired 95% CI {ci['lo']:+.4f},{ci['hi']:+.4f}; MDE {ci['mde']:.4f}; "
                 f"n={ci['n']}].\n")
    verdict = ("ADAPTING BEATS the fixed order (CI excludes 0)" if ci["lo"] > 0 else
               "ADAPTING LOSES to the fixed order (CI excludes 0)" if ci["hi"] < 0 else
               "TIE within noise (CI spans 0)")
    lines.append(f"\n**Verdict (turn 8): {verdict}.** (DEV/synthetic only; the 173/300 real users "
                 "are out of bounds.)\n")
    open(MD, "a", encoding="utf-8").write("".join(lines))

    print("\n" + "=" * 64, flush=True)
    print("ADAPTIVE EIG vs STATIC (TEST subsample, per-turn NDCG@10)", flush=True)
    print(f"  COLD(t0) = {cold0:.4f}   n={len(test_sub)}", flush=True)
    print(f"  {'turn':>4} {'static-ig':>10} {'adaptive':>10} {'delta':>9}", flush=True)
    for tt in range(1, Tmax + 1):
        print(f"  {tt:>4} {static_ig[tt]:>10.4f} {adaptive[tt]:>10.4f} "
              f"{adaptive[tt]-static_ig[tt]:>+9.4f}{'  <-turn8' if tt == 8 else ''}", flush=True)
    print(f"\n  turn-8 adaptive-minus-static = {d8:+.4f} "
          f"[CI {ci['lo']:+.4f},{ci['hi']:+.4f} MDE {ci['mde']:.4f} n={ci['n']}]", flush=True)
    print(f"  VERDICT: {verdict}", flush=True)
    print(f"\n[done] wrote {AC.CACHE_DIR}/adaptive_eig.json + {MD} "
          f"[{(time.time()-t0)/60:.1f} min]", flush=True)


if __name__ == "__main__":
    main()
