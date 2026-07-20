"""arena_b2_budget.py -- B2 PER-TURN @10 budget-finding rerun (author directive 2026-07-10).

Recompute the LEARNED STATIC baseline (b2 = greedy forward selection maximising DEV-cohort
NDCG@10) and produce its full per-turn curve out to 25 turns, so the author can pick a realistic
interview cutoff. Same gated v2.1 world + fold-v3 the prior arena used (comparable). DEV synthetic
ONLY; $0; no LLM; the 173/300 real-judged users are NEVER touched. Deterministic (seed 123).

Deliverable: NDCG@10 at every turn 0..25, marginal gain per turn, plus two anchors on the same
cohort/metric -- turn 0 cold-start, and the full-profile ceiling (fold ALL of the user's real
known info). One-line cutoff read appended to experiments/ARENA_BUILD.md.

Reuses the running learnability gate's shared prescreen cache (pres31_b2_n300_K10.json) and, if
present, seeds the T25 greedy from its completed T24 sequence (greedy is prefix-deterministic:
the 25-turn sequence is the 24-turn one + 1 more greedy pick).
"""
import os, sys, json, time, shutil
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE); sys.path.insert(0, os.path.join(_HERE, "instrument2"))
import warnings
warnings.filterwarnings("ignore")

import arena_core as AC
import arena_policies as AP
from arena_core import ndcg_at_k_batch
from arena_policies import StaticSeq, run_policy

MD = f"{AC.CACHE_DIR}/b2_budget_section.md"   # sidecar; agent appends to ARENA_BUILD.md after
CACHE = AC.CACHE_DIR
Tmax = 25                    # extend to 25 turns (author budget-finding ask)
K = 10                       # PRIMARY metric = NDCG@10
KS = (10, 50)                # @50 secondary
N_DEVTEST = 600              # enlarged DEV-TEST cohort (the current arena primary; same world)


def md(t):
    open(MD, "a", encoding="utf-8").write(t)


# fresh sidecar each run
if os.path.exists(MD):
    os.remove(MD)


def main():
    t0 = time.time()
    ar = AC.Arena()
    coh = AC.make_cohorts(ar, n_train=1000, n_devval=80, n_devtest=N_DEVTEST)
    cfg = coh["cfg"]
    ar.prefill_answers(coh["train"], "train", cfg, verbose=False)
    ar.prefill_answers(coh["devtest"], "devtest_big", cfg, verbose=False)
    ar.set_pop_prior(AP.train_pop_prior(ar, coh["train"]))
    dt = coh["devtest"]
    b2_users = coh["train"][:300]

    # --- seed T25 greedy cache from the learnability gate's completed T24 sequence (if present) ---
    p24 = f"{CACHE}/b2v31_T24_K10_n300.json"
    p25 = f"{CACHE}/b2v31_T25_K10_n300.json"
    if os.path.exists(p24) and not os.path.exists(p25):
        blob = json.load(open(p24))
        if len(blob["seq"]) >= 24:
            shutil.copy(p24, p25)
            print(f"[b2budget] seeded T25 greedy from completed T24 ({len(blob['seq'])} picks)",
                  flush=True)

    # --- build b2 = greedy forward selection on endpoint NDCG@10, out to 25 turns ---
    print("[b2budget] building/extending b2 greedy on NDCG@10 to T=25 ...", flush=True)
    b2_seq, b2_gain = AP.build_b2(ar, b2_users, Tmax=Tmax, K=K, prescreen_top=300, verbose=True)

    # --- per-turn curve on the DEV-TEST cohort (turn 0 = cold-start anchor) ---
    print(f"[b2budget] evaluating b2 per-turn curve on {len(dt)} DEV-TEST users ...", flush=True)
    o = run_policy(ar, dt, StaticSeq("b2", b2_seq), Tmax, Ks=KS, tag="b2budget", verbose=True)
    curve10 = np.nanmean(o["curves"][10], axis=0)     # len Tmax+1 (0..25)
    curve50 = np.nanmean(o["curves"][50], axis=0)
    cold10 = float(curve10[0])

    # --- full-profile ceiling: fold ALL of the user's real known info (every universe question) ---
    print("[b2budget] full-profile ceiling (fold all known info per user) ...", flush=True)
    ctxs = [ar.user_ctx(r) for r in dt]
    held = [r["held"] for r in dt]; prof = [set(r["known"].keys()) for r in dt]
    tok_lists = []; nat_lists = []
    for i, r in enumerate(dt):
        uid = r["u"]; ctx = ctxs[i]
        toks = []; nat = []
        for qi in range(ar.nQ):
            toks += ar.tokens_for(uid, qi, ctx)
            if ar.is_liked_item(uid, qi, ctx) and ar.answered(uid, qi, ctx):
                nat.append(int(ar.uni.bank[qi - ar.off_item]))
        tok_lists.append(toks); nat_lists.append(nat)
    Zfull = ar.belief_z_batch(tok_lists, nat_lists)
    full10 = float(np.nanmean([v if v is not None else np.nan
                               for v in ndcg_at_k_batch(ar.FR, Zfull, held, prof, 10)]))

    # --- assemble table ---
    rows = []
    for t in range(1, Tmax + 1):
        nd = float(curve10[t]); prev = float(curve10[t - 1])
        rows.append((t, nd, nd - prev))
    gap_cold_full = full10 - cold10

    # cutoff read: where marginal gain drops below a noise floor (approx per-turn granularity ~1e-3)
    NOISE = 0.001
    below = [t for (t, nd, mg) in rows if mg < NOISE]
    first_below = below[0] if below else None
    # where curve reaches ~95% of the cold->full closure
    closes = [t for (t, nd, mg) in rows if (nd - cold10) >= 0.95 * gap_cold_full] if gap_cold_full > 0 else []
    close95 = closes[0] if closes else None

    # --- console report ---
    print("\n[b2budget] PER-TURN NDCG@10 (b2 greedy static, DEV-TEST n=%d)" % len(dt), flush=True)
    print(f"  turn 0 (cold-start): {cold10:.4f}", flush=True)
    for (t, nd, mg) in rows:
        print(f"  turn {t:2d}: {nd:.4f}   marg {mg:+.4f}", flush=True)
    print(f"  full-profile ceiling: {full10:.4f}  (cold->full gap {gap_cold_full:+.4f})", flush=True)
    print(f"  first turn with marginal gain < {NOISE}: {first_below}", flush=True)
    print(f"  first turn reaching 95% of cold->full closure: {close95}", flush=True)

    # --- persist results ---
    json.dump(dict(primary="ndcg@10", n_devtest=len(dt), Tmax=Tmax,
                   b2_seq=[int(q) for q in b2_seq], b2_seq_names=[ar.q_names[int(q)] for q in b2_seq],
                   greedy_train_gain=[float(g) for g in b2_gain],
                   curve10=[float(x) for x in curve10], curve50=[float(x) for x in curve50],
                   cold10=cold10, full_profile10=full10,
                   marginal10=[float(mg) for (_, _, mg) in rows],
                   first_below_noise=first_below, close95_turn=close95),
              open(f"{CACHE}/b2_budget_at10.json", "w"), indent=1)

    # --- append MD section ---
    md("\n\n---\n\n## B2 PER-TURN @10 (budget-finding rerun)\n\n")
    md("Author directive 2026-07-10: recompute the learned static baseline **b2** (greedy forward "
       "selection maximising DEV-cohort **NDCG@10**) and give its full per-turn curve out to **25 "
       "turns**, to pick a realistic interview cutoff. SAME gated v2.1 world (fitted models sha "
       "f53e23a7692d + EASE backbone), SAME fold-v3 belief (val 0.4356), SAME 2,428-question "
       "universe, SAME DEV construction (greedy on 300 TRAIN users, prescreen top-300), lenient "
       "regime. DEV SYNTHETIC ONLY (n=%d DEV-TEST); the 173/300 real-judged users are untouched. "
       "Deterministic (seed 123). NO LLM, $0.\n\n" % len(dt))
    md("Anchors on the SAME cohort/metric: **turn 0 = cold-start** NDCG@10 = **%.4f**; "
       "**full-profile ceiling** (fold ALL of each user's real known info over the whole universe) "
       "NDCG@10 = **%.4f** (cold->full headroom **%+.4f**).\n\n" % (cold10, full10, gap_cold_full))
    md("| turn | NDCG@10 | marginal gain vs prev |\n|--:|--:|--:|\n")
    md("| 0 (cold) | %.4f | -- |\n" % cold10)
    for (t, nd, mg) in rows:
        md("| %d | %.4f | %+.4f |\n" % (t, nd, mg))
    md("| ceiling (full profile) | %.4f | -- |\n\n" % full10)
    md("**Read:** first turn whose marginal gain falls below the ~%.3f per-turn granularity floor: "
       "**T=%s**; first turn reaching 95%% of the cold->full-profile closure: **T=%s**. "
       "Candidate interview cutoffs sit in that band.\n\n" % (NOISE, str(first_below), str(close95)))
    md("### b2's decoded schedule (25 greedy picks, in order)\n\n")
    for i, qi in enumerate(b2_seq):
        md("%d. %s\n" % (i + 1, ar.q_names[int(qi)]))
    print(f"[b2budget] DONE [{(time.time()-t0)/60:.1f}m]; appended to {MD}", flush=True)


if __name__ == "__main__":
    main()
